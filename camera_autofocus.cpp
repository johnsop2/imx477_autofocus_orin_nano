#include <cuda_runtime.h>

#include <gst/gst.h>
#include <gst/app/gstappsink.h>
#include <gst/video/video.h>

#include <linux/i2c-dev.h>

#include <sys/ioctl.h>
#include <sys/types.h>
#include <sys/wait.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <deque>
#include <fcntl.h>
#include <format>
#include <iostream>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <unistd.h>
#include <vector>

#include "laplacian.h"

// ============================================================
// GLOBAL STATE
// ============================================================

static std::atomic<bool> g_running{true};


// ============================================================
// SIGNAL HANDLING
// ============================================================

static void signal_handler(int)
{
    g_running.store(false, std::memory_order_relaxed);
}


// ============================================================
// I2C FOCUS CONTROLLER
// ============================================================

class FocusController
{
public:
    FocusController() = default;

    ~FocusController()
    {
        close();
    }

    bool open_device(
        const char* device,
        int address)
    {
        fd_ = ::open(
            device,
            O_RDWR);

        if (fd_ < 0)
        {
            std::perror(
                "Failed to open I2C device");

            return false;
        }

        if (ioctl(
                fd_,
                I2C_SLAVE,
                address) < 0)
        {
            std::perror(
                "Failed to select I2C slave");

            close();

            return false;
        }

        std::cout
            << "I2C focus controller: "
            << device
            << ", address 0x"
            << std::hex
            << address
            << std::dec
            << '\n';

        return true;
    }

    void close()
    {
        if (fd_ >= 0)
        {
            ::close(fd_);
            fd_ = -1;
        }
    }

    bool set_focus(uint16_t value)
    {
        if (fd_ < 0)
        {
            std::cerr
                << "I2C device is not open.\n";

            return false;
        }

        union {
        uint16_t data;
        uint8_t array[2];
        } buffer;

   // Lens controller accepts a 10-bit value.
        value &= 0x03FF;
        buffer.data = (value >> 8) | (value << 8);

        while (true)
        {
            const ssize_t result =
                ::write(
                    fd_,
                    buffer.array,
                    sizeof(buffer));

            if (result == sizeof(buffer))
                return true;

            if (result < 0 &&
                errno == EINTR)
            {
                continue;
            }

            std::perror(
                "I2C focus write failed");

            return false;
        }
    }

private:
    int fd_ = -1;
};


// ============================================================
// CONFIGURATION
// ============================================================

struct Config
{
    int camera = 0;
    int width = 1920;
    int height = 1080;
    int fps = 30;

    std::string alsa_device = "default";

    std::string output =
        "/home/pj/Videos/" +
        std::format(
            "{:%Y-%m-%d-%H-%M-%S}",
            std::chrono::system_clock::now()) +
        ".mkv";

    // --------------------------------------------------------
    // OUTPUT OPTIONS
    // --------------------------------------------------------
    //
    // Defaults preserve the original behavior.
    //
    bool record = true;
    bool stream = true;
    bool display = true;
};


// ============================================================
// INTEGER PARSING
// ============================================================

static std::optional<int> parse_int(
    std::string_view text)
{
    int value = 0;

    const char* first =
        text.data();

    const char* last =
        text.data() +
        text.size();

    const auto result =
        std::from_chars(
            first,
            last,
            value);

    if (result.ec != std::errc{} ||
        result.ptr != last)
    {
        return std::nullopt;
    }

    return value;
}


// ============================================================
// ARGUMENT PARSING
// ============================================================

static void usage(
    const char* name)
{
    std::cout
        << "Usage: "
        << name
        << " [options]\n\n"

        << "Camera options:\n"
        << "  -c N          Camera ID (default 0)\n"
        << "  -w N          Width (default 1920)\n"
        << "  -h N          Height (default 1080)\n"
        << "  -f N          FPS (default 30)\n"
        << "  -a DEVICE     ALSA device (default default)\n"
        << "  -o FILE       Output MKV filename\n\n"

        << "Output options:\n"
        << "  --record      Enable MKV recording (default)\n"
        << "  --no-record   Disable MKV recording\n"
        << "  --stream      Enable VLC UDP streaming (default)\n"
        << "  --no-stream   Disable VLC UDP streaming\n"
        << "  --display      Enable local display (default)\n"
        << "  --no-display   Disable local display\n\n"

        << "  --help        Show this help\n\n"

        << "Examples:\n"
        << "  " << name << "\n"
        << "      Record + VLC stream + local display\n\n"

        << "  " << name << " --no-display\n"
        << "      Record + VLC stream, no local display\n\n"

        << "  " << name << " --no-stream\n"
        << "      Record + local display, no VLC stream\n\n"

        << "  " << name << " --no-record --no-display\n"
        << "      Autofocus + VLC stream only\n\n"

        << "  " << name << " --no-record --no-stream\n"
        << "      Autofocus + local display only\n\n"

        << "  " << name << " --no-record --no-stream --no-display\n"
        << "      Autofocus only\n";
}


static bool parse_args(
    int argc,
    char** argv,
    Config& config,
    bool& show_help)
{
    show_help = false;

    for (int i = 1;
         i < argc;
         ++i)
    {
        const std::string arg =
            argv[i];

        if (arg == "--help")
        {
            show_help = true;
            return true;
        }

        // ----------------------------------------------------
        // ENABLE / DISABLE OPTIONS
        // ----------------------------------------------------

        if (arg == "--record")
        {
            config.record = true;
            continue;
        }

        if (arg == "--no-record")
        {
            config.record = false;
            continue;
        }

        if (arg == "--stream")
        {
            config.stream = true;
            continue;
        }

        if (arg == "--no-stream")
        {
            config.stream = false;
            continue;
        }

        if (arg == "--display")
        {
            config.display = true;
            continue;
        }

        if (arg == "--no-display")
        {
            config.display = false;
            continue;
        }

        // ----------------------------------------------------
        // STRING OPTIONS
        // ----------------------------------------------------

        if (arg == "-o" ||
            arg == "-a")
        {
            if (i + 1 >= argc)
            {
                std::cerr
                    << "Missing value for "
                    << arg
                    << '\n';

                return false;
            }

            const std::string value =
                argv[++i];

            if (arg == "-o")
                config.output = value;
            else
                config.alsa_device = value;

            continue;
        }

        // ----------------------------------------------------
        // INTEGER OPTIONS
        // ----------------------------------------------------

        if (arg == "-c" ||
            arg == "-w" ||
            arg == "-h" ||
            arg == "-f")
        {
            if (i + 1 >= argc)
            {
                std::cerr
                    << "Missing value for "
                    << arg
                    << '\n';

                return false;
            }

            const auto value =
                parse_int(argv[++i]);

            if (!value)
            {
                std::cerr
                    << "Invalid integer for "
                    << arg
                    << '\n';

                return false;
            }

            if (arg == "-c")
                config.camera = *value;
            else if (arg == "-w")
                config.width = *value;
            else if (arg == "-h")
                config.height = *value;
            else
                config.fps = *value;

            continue;
        }

        std::cerr
            << "Unknown argument: "
            << arg
            << '\n';

        return false;
    }

    if (config.camera < 0)
    {
        std::cerr
            << "Camera ID cannot be negative.\n";

        return false;
    }

    if (config.width < 3 ||
        config.height < 3 ||
        config.fps <= 0)
    {
        std::cerr
            << "Invalid resolution or FPS.\n";

        return false;
    }

    // YUV422 has half horizontal chroma resolution.
    if (config.width % 2 != 0)
    {
        std::cerr
            << "YUV422 requires an even width.\n";

        return false;
    }

    if (config.alsa_device.empty())
    {
        std::cerr
            << "ALSA device cannot be empty.\n";

        return false;
    }

    if (config.record &&
        config.output.empty())
    {
        std::cerr
            << "Output filename cannot be empty when "
               "recording is enabled.\n";

        return false;
    }

    return true;
}


// ============================================================
// YUV FRAME
// ============================================================

struct YuvFrame
{
    std::vector<uint8_t> data;
};


// ============================================================
// PLANE COPY
// ============================================================

static void copy_plane(
    uint8_t* destination,
    const guint8* source,
    gint source_stride,
    gint width,
    gint height)
{
    for (gint row = 0;
         row < height;
         ++row)
    {
        std::memcpy(
            destination +
                static_cast<size_t>(row) *
                width,

            source +
                static_cast<size_t>(row) *
                source_stride,

            static_cast<size_t>(width));
    }
}


// ============================================================
// VLC STREAM
// ============================================================

constexpr const char* VLC_STREAM_HOST = "0.0.0.0";
constexpr int VLC_STREAM_PORT = 1234;


// ============================================================
// FFMPEG WRITER
// ============================================================
//
// This class can independently:
//
//   1. Record MKV
//   2. Stream MPEG-TS to VLC
//   3. Do both
//
// If neither recording nor streaming is enabled, this class
// is not started.
//
// ============================================================

class FfmpegWriter
{
public:
    FfmpegWriter(
        std::string filename,
        int width,
        int height,
        int fps,
        std::string alsa_device,
        bool record,
        bool stream,
        size_t max_queue_size = 8)
        :
        filename_(std::move(filename)),
        width_(width),
        height_(height),
        fps_(fps),
        alsa_device_(std::move(alsa_device)),
        record_(record),
        stream_(stream),
        max_queue_size_(max_queue_size)
    {
    }

    ~FfmpegWriter()
    {
        stop();
    }

    FfmpegWriter(
        const FfmpegWriter&) = delete;

    FfmpegWriter& operator=(
        const FfmpegWriter&) = delete;

    bool enabled() const
    {
        return record_ || stream_;
    }

    bool start()
    {
        if (!enabled())
            return true;

        if (running_)
            return false;

        int pipe_fds[2];

        if (pipe(pipe_fds) != 0)
        {
            std::perror(
                "pipe");

            return false;
        }

        ffmpeg_pid_ =
            fork();

        if (ffmpeg_pid_ < 0)
        {
            std::perror(
                "fork");

            close(pipe_fds[0]);
            close(pipe_fds[1]);

            return false;
        }

        if (ffmpeg_pid_ == 0)
        {
            // ------------------------------------------------
            // CHILD
            // ------------------------------------------------

            if (dup2(
                    pipe_fds[0],
                    STDIN_FILENO) < 0)
            {
                std::perror(
                    "dup2");

                _exit(127);
            }

            close(pipe_fds[0]);
            close(pipe_fds[1]);

            const std::string resolution =
                std::to_string(width_) +
                "x" +
                std::to_string(height_);

            const std::string framerate =
                std::to_string(fps_);

            // ------------------------------------------------
            // BUILD OUTPUT LIST
            // ------------------------------------------------
            //
            // FFmpeg tee syntax:
            //
            //   [f=matroska]file.mkv
            //   [f=mpegts]udp://0.0.0.0:1234
            //
            // Only requested outputs are included.
            // ------------------------------------------------

            std::string tee_output;

            if (record_)
            {
                tee_output =
                    "[f=matroska]" +
                    filename_;
            }

            if (stream_)
            {
                if (!tee_output.empty())
                    tee_output += "|";

                tee_output +=
                    "[f=mpegts]udp://" +
                    std::string(VLC_STREAM_HOST) +
                    ":" +
                    std::to_string(VLC_STREAM_PORT) +
                    "?pkt_size=1316";
            }

            // ------------------------------------------------
            // AUDIO IS REQUIRED ONLY FOR RECORDING/STREAMING
            // ------------------------------------------------

            execlp(
                "ffmpeg",
                "ffmpeg",

                "-hide_banner",
                "-loglevel", "warning",
                "-y",

                // ------------------------------------------------
                // RAW VIDEO
                // ------------------------------------------------

                "-f", "rawvideo",
                "-pix_fmt", "yuv422p",
                "-video_size", resolution.c_str(),
                "-framerate", framerate.c_str(),
                "-i", "pipe:0",

                // ------------------------------------------------
                // AUDIO
                // ------------------------------------------------

                "-f", "alsa",
                "-i", alsa_device_.c_str(),

                // ------------------------------------------------
                // VIDEO
                // ------------------------------------------------

                "-map", "0:v:0",
                "-c:v", "mpeg4",
                "-pix_fmt", "yuv422p",

                // ------------------------------------------------
                // AUDIO
                // ------------------------------------------------

                "-map", "1:a:0",
                "-c:a", "aac",
                "-b:a", "192k",
                "-ar", "48000",
                "-ac", "2",

                "-shortest",

                // ------------------------------------------------
                // REQUESTED OUTPUT(S)
                // ------------------------------------------------

                "-f", "tee",
                tee_output.c_str(),

                static_cast<char*>(nullptr));

            std::perror(
                "execlp ffmpeg");

            _exit(127);
        }

        // ----------------------------------------------------
        // PARENT
        // ----------------------------------------------------

        close(pipe_fds[0]);

        ffmpeg_stdin_ =
            pipe_fds[1];

        {
            std::lock_guard lock(mutex_);

            accepting_ = true;
            compression_error_ = false;
            running_ = true;
        }

        compression_thread_ =
            std::thread(
                &FfmpegWriter::compression_loop,
                this);

        std::cout
            << "FFmpeg started. PID="
            << ffmpeg_pid_
            << '\n';

        if (record_)
        {
            std::cout
                << "Recording:   "
                << filename_
                << '\n';
        }
        else
        {
            std::cout
                << "Recording:   DISABLED\n";
        }

        if (stream_)
        {
            std::cout
                << "VLC stream:  udp://@:"
                << VLC_STREAM_PORT
                << " (MPEG-TS)\n";
        }
        else
        {
            std::cout
                << "VLC stream:  DISABLED\n";
        }

        return true;
    }

    bool push(YuvFrame&& frame)
    {
        if (!enabled())
            return true;

        std::unique_lock lock(mutex_);

        cv_space_.wait(
            lock,
            [this]()
            {
                return queue_.size() <
                           max_queue_size_ ||
                       !accepting_ ||
                       compression_error_;
            });

        if (!accepting_ ||
            compression_error_)
        {
            return false;
        }

        queue_.emplace_back(
            std::move(frame));

        cv_data_.notify_one();

        return true;
    }

    void stop()
    {
        if (!enabled())
            return;

        {
            std::lock_guard lock(mutex_);

            accepting_ = false;
        }

        cv_data_.notify_all();
        cv_space_.notify_all();

        if (compression_thread_.joinable())
        {
            compression_thread_.join();
        }

        if (ffmpeg_stdin_ >= 0)
        {
            close(ffmpeg_stdin_);
            ffmpeg_stdin_ = -1;
        }

        wait_for_ffmpeg();
    }

    bool has_error() const
    {
        std::lock_guard lock(mutex_);
        return compression_error_;
    }

    size_t queue_size() const
    {
        std::lock_guard lock(mutex_);
        return queue_.size();
    }

private:
    void set_error()
    {
        {
            std::lock_guard lock(mutex_);

            compression_error_ = true;
            accepting_ = false;
        }

        cv_data_.notify_all();
        cv_space_.notify_all();
    }

    void compression_loop()
    {
        while (true)
        {
            YuvFrame frame;

            {
                std::unique_lock lock(mutex_);

                cv_data_.wait(
                    lock,
                    [this]()
                    {
                        return !queue_.empty() ||
                               !accepting_;
                    });

                if (queue_.empty() &&
                    !accepting_)
                {
                    break;
                }

                frame =
                    std::move(
                        queue_.front());

                queue_.pop_front();

                cv_space_.notify_one();
            }

            if (frame.data.empty())
                continue;

            const uint8_t* ptr =
                frame.data.data();

            size_t remaining =
                frame.data.size();

            while (remaining > 0)
            {
                const ssize_t result =
                    ::write(
                        ffmpeg_stdin_,
                        ptr,
                        remaining);

                if (result > 0)
                {
                    ptr += result;

                    remaining -=
                        static_cast<size_t>(
                            result);

                    continue;
                }

                if (result < 0 &&
                    errno == EINTR)
                {
                    continue;
                }

                if (result < 0)
                {
                    std::perror(
                        "FFmpeg pipe write");

                    set_error();

                    return;
                }

                std::cerr
                    << "FFmpeg pipe closed.\n";

                set_error();

                return;
            }
        }
    }

    void wait_for_ffmpeg()
    {
        if (ffmpeg_pid_ <= 0)
            return;

        int status = 0;

        while (true)
        {
            const pid_t result =
                waitpid(
                    ffmpeg_pid_,
                    &status,
                    0);

            if (result >= 0)
                break;

            if (errno == EINTR)
                continue;

            std::perror(
                "waitpid");

            set_error();

            break;
        }

        if (WIFEXITED(status))
        {
            const int exit_code =
                WEXITSTATUS(status);

            if (exit_code != 0)
            {
                std::cerr
                    << "FFmpeg exited with code "
                    << exit_code
                    << '\n';

                set_error();
            }
        }
        else if (WIFSIGNALED(status))
        {
            std::cerr
                << "FFmpeg terminated by signal "
                << WTERMSIG(status)
                << '\n';

            set_error();
        }

        ffmpeg_pid_ = -1;
        running_ = false;
    }

private:
    std::string filename_;

    int width_;
    int height_;
    int fps_;

    std::string alsa_device_;

    bool record_;
    bool stream_;

    size_t max_queue_size_;

    pid_t ffmpeg_pid_ = -1;

    int ffmpeg_stdin_ = -1;

    std::deque<YuvFrame> queue_;

    mutable std::mutex mutex_;

    std::condition_variable cv_data_;
    std::condition_variable cv_space_;

    std::thread compression_thread_;

    bool accepting_ = false;
    bool compression_error_ = false;
    bool running_ = false;
};


// ============================================================
// GSTREAMER SAMPLE VALIDATION
// ============================================================

static bool get_video_frame(
    GstSample* sample,
    GstVideoInfo& info,
    GstVideoFrame& frame)
{
    if (!sample)
        return false;

    GstCaps* caps =
        gst_sample_get_caps(sample);

    GstBuffer* buffer =
        gst_sample_get_buffer(sample);

    if (!caps || !buffer)
        return false;

    if (!gst_video_info_from_caps(
            &info,
            caps))
    {
        return false;
    }

    if (GST_VIDEO_INFO_FORMAT(&info) !=
        GST_VIDEO_FORMAT_Y42B)
    {
        std::cerr
            << "Unexpected format: "
            << GST_VIDEO_INFO_NAME(&info)
            << '\n';

        return false;
    }

    if (!gst_video_frame_map(
            &frame,
            &info,
            buffer,
            GST_MAP_READ))
    {
        return false;
    }

    return true;
}


// ============================================================
// AUTOFOCUS SEARCH CONTROLLER
// ============================================================
//
// The controller searches the lens position in progressively
// smaller steps. Each position is measured over several frames;
// the first frame after a lens move is discarded as a settling
// frame. Five measurements are reduced with a trimmed mean.
// A three-point parabolic estimate is used to refine the peak.
// ============================================================

class AutofocusController
{
public:
    static constexpr int FOCUS_MIN = 10;
    static constexpr int FOCUS_MAX = 1000;

    bool start(FocusController& focus)
    {
        stage_index_ = 0;
        finished_ = false;
        failed_ = false;
        moving_ = true;
        position_index_ = 0;
        settle_remaining_ = 0;
        samples_.clear();
        measurements_.clear();
        best_position_ = FOCUS_MIN;
        best_score_ = 0.0;
        peak_estimate_ =
            static_cast<double>(FOCUS_MIN);

        return begin_stage(
            focus,
            FOCUS_MIN);
    }

    bool process(
        FocusController& focus,
        CudaFocusProcessor& metric,
        const uint8_t* y,
        size_t stride)
    {
        if (finished_ || failed_)
            return !failed_;

        if (moving_)
        {
            moving_ = false;
            settle_remaining_ =
                SETTLE_FRAMES;

            samples_.clear();
        }

        if (settle_remaining_ > 0)
        {
            --settle_remaining_;
            return true;
        }

        double score = 0.0;

        if (!metric.calculate(
                y,
                stride,
                score))
        {
            std::cerr
                << "CUDA focus calculation failed.\n";

            failed_ = true;

            return false;
        }

        samples_.push_back(score);

        std::cout
            << "\rAF stage "
            << (stage_index_ + 1)
            << "/"
            << STAGE_COUNT
            << "  pos "
            << current_position_
            << "  frame "
            << samples_.size()
            << "/"
            << FRAMES_PER_POSITION
            << "  score "
            << score
            << "          "
            << std::flush;

        if (samples_.size() <
            FRAMES_PER_POSITION)
        {
            return true;
        }

        const double averaged_score =
            trimmed_mean(samples_);

        measurements_.push_back(
            Measurement{
                current_position_,
                averaged_score});

        samples_.clear();

        ++position_index_;

        if (position_index_ <
            positions_.size())
        {
            return move_to_current_position(
                focus);
        }

        return finish_stage(focus);
    }

    bool finished() const
    {
        return finished_;
    }

    bool failed() const
    {
        return failed_;
    }

private:
    struct Measurement
    {
        int position;
        double score;
    };

    static constexpr int FRAMES_PER_POSITION = 5;
    static constexpr int SETTLE_FRAMES = 1;
    static constexpr int STAGE_COUNT = 6;

    static constexpr int STEPS[STAGE_COUNT] =
    {
        40, 20, 10, 5, 2, 1
    };

    int stage_index_ = 0;
    size_t position_index_ = 0;

    int current_position_ =
        FOCUS_MIN;

    bool moving_ = false;
    bool finished_ = false;
    bool failed_ = false;

    int settle_remaining_ = 0;

    std::vector<int> positions_;
    std::vector<double> samples_;
    std::vector<Measurement> measurements_;

    int best_position_ =
        FOCUS_MIN;

    double best_score_ = 0.0;

    double peak_estimate_ =
        FOCUS_MIN;

    static int clamp_position(
        int position)
    {
        if (position < FOCUS_MIN)
            return FOCUS_MIN;

        if (position > FOCUS_MAX)
            return FOCUS_MAX;

        return position;
    }

    static double trimmed_mean(
        const std::vector<double>& values)
    {
        if (values.empty())
            return 0.0;

        std::vector<double> sorted =
            values;

        std::sort(
            sorted.begin(),
            sorted.end());

        if (sorted.size() < 3)
        {
            double sum = 0.0;

            for (double value : sorted)
                sum += value;

            return sum /
                   static_cast<double>(
                       sorted.size());
        }

        double sum = 0.0;

        for (size_t i = 1;
             i + 1 < sorted.size();
             ++i)
        {
            sum += sorted[i];
        }

        return sum /
               static_cast<double>(
                   sorted.size() - 2);
    }

    bool begin_stage(
        FocusController& focus,
        int center)
    {
        const int step =
            STEPS[stage_index_];

        int half_window;

        if (stage_index_ == 0)
            half_window =
                FOCUS_MAX - FOCUS_MIN;
        else
            half_window =
                step * 2;

        positions_.clear();

        const int first =
            clamp_position(
                center - half_window);

        const int last =
            clamp_position(
                center + half_window);

        for (int position = first;
             position <= last;
             position += step)
        {
            positions_.push_back(
                position);
        }

        if (positions_.empty() ||
            positions_.back() != last)
        {
            positions_.push_back(last);
        }

        std::sort(
            positions_.begin(),
            positions_.end());

        positions_.erase(
            std::unique(
                positions_.begin(),
                positions_.end()),
            positions_.end());

        position_index_ = 0;

        measurements_.clear();
        samples_.clear();

        return move_to_current_position(
            focus);
    }

    bool move_to_current_position(
        FocusController& focus)
    {
        if (position_index_ >=
            positions_.size())
        {
            failed_ = true;
            return false;
        }

        current_position_ =
            positions_[position_index_];

        if (!focus.set_focus(
                static_cast<uint16_t>(
                    current_position_)))
        {
            std::cerr
                << "\nFailed to set focus position "
                << current_position_
                << ".\n";

            failed_ = true;

            return false;
        }

        moving_ = true;

        return true;
    }

    double interpolate_peak(
        size_t best_index) const
    {
        if (best_index == 0 ||
            best_index + 1 >=
                measurements_.size())
        {
            return static_cast<double>(
                measurements_[best_index].position);
        }

        const auto& left =
            measurements_[best_index - 1];

        const auto& center =
            measurements_[best_index];

        const auto& right =
            measurements_[best_index + 1];

        const double denominator =
            left.score -
            2.0 * center.score +
            right.score;

        if (std::abs(denominator) <
            1.0e-12)
        {
            return static_cast<double>(
                center.position);
        }

        const double offset =
            0.5 *
            (left.score - right.score) /
            denominator;

        const double step =
            static_cast<double>(
                right.position -
                center.position);

        const double limited_offset =
            std::clamp(
                offset,
                -1.0,
                1.0);

        return static_cast<double>(
                   center.position) +
               limited_offset * step;
    }

    bool finish_stage(
        FocusController& focus)
    {
        if (measurements_.empty())
        {
            failed_ = true;
            return false;
        }

        size_t best_index = 0;

        for (size_t i = 1;
             i < measurements_.size();
             ++i)
        {
            if (measurements_[i].score >
                measurements_[best_index].score)
            {
                best_index = i;
            }
        }

        best_position_ =
            measurements_[best_index].position;

        best_score_ =
            measurements_[best_index].score;

        peak_estimate_ =
            interpolate_peak(best_index);

        std::cout
            << "\nStage "
            << (stage_index_ + 1)
            << " complete: best="
            << best_position_
            << "  score="
            << best_score_
            << "  interpolated peak="
            << peak_estimate_
            << '\n';

        if (stage_index_ + 1 >=
            STAGE_COUNT)
        {
            const int final_position =
                clamp_position(
                    static_cast<int>(
                        std::lround(
                            peak_estimate_)));

            if (!focus.set_focus(
                    static_cast<uint16_t>(
                        final_position)))
            {
                std::cerr
                    << "Failed to move to final focus.\n";

                failed_ = true;

                return false;
            }

            best_position_ =
                final_position;

            finished_ = true;

            std::cout
                << "\nFocus locked at position "
                << best_position_
                << "  score="
                << best_score_
                << '\n';

            return true;
        }

        ++stage_index_;

        const int next_center =
            clamp_position(
                static_cast<int>(
                    std::lround(
                        peak_estimate_)));

        return begin_stage(
            focus,
            next_center);
    }
};

constexpr int AutofocusController::STEPS[
    AutofocusController::STAGE_COUNT];


// ============================================================
// MAIN
// ============================================================

int main(
    int argc,
    char** argv)
{
    // --------------------------------------------------------
    // SIGPIPE
    // --------------------------------------------------------

    struct sigaction pipe_action{};

    pipe_action.sa_handler =
        SIG_IGN;

    sigemptyset(
        &pipe_action.sa_mask);

    sigaction(
        SIGPIPE,
        &pipe_action,
        nullptr);


    // --------------------------------------------------------
    // SIGINT / SIGTERM
    // --------------------------------------------------------

    struct sigaction stop_action{};

    stop_action.sa_handler =
        signal_handler;

    sigemptyset(
        &stop_action.sa_mask);

    sigaction(
        SIGINT,
        &stop_action,
        nullptr);

    sigaction(
        SIGTERM,
        &stop_action,
        nullptr);


    // --------------------------------------------------------
    // COMMAND LINE
    // --------------------------------------------------------

    Config config;
    bool show_help = false;

    if (!parse_args(
            argc,
            argv,
            config,
            show_help))
    {
        usage(argv[0]);

        return EXIT_FAILURE;
    }

    if (show_help)
    {
        usage(argv[0]);

        return EXIT_SUCCESS;
    }


    // --------------------------------------------------------
    // CONFIGURATION STATUS
    // --------------------------------------------------------

    std::cout
        << "\n========================================\n"
        << " Camera Autofocus\n"
        << "========================================\n"
        << "Camera:       "
        << config.camera
        << '\n'
        << "Resolution:   "
        << config.width
        << " x "
        << config.height
        << '\n'
        << "FPS:          "
        << config.fps
        << '\n'
        << "Recording:    "
        << (config.record ? "ENABLED" : "DISABLED")
        << '\n'
        << "VLC stream:   "
        << (config.stream ? "ENABLED" : "DISABLED")
        << '\n'
        << "Display:      "
        << (config.display ? "ENABLED" : "DISABLED")
        << '\n';

    if (config.record)
    {
        std::cout
            << "Output:       "
            << config.output
            << '\n';
    }

    if (config.stream)
    {
        std::cout
            << "VLC URL:      udp://@: "
            << VLC_STREAM_PORT
            << '\n';
    }

    std::cout
        << "========================================\n\n";


    // --------------------------------------------------------
    // I2C
    // --------------------------------------------------------

    FocusController focus_controller;

    if (!focus_controller.open_device(
            "/dev/i2c-10",
            0x0C))
    {
        return EXIT_FAILURE;
    }


    // --------------------------------------------------------
    // CUDA
    // --------------------------------------------------------

    CudaFocusProcessor cuda_focus;

    if (!cuda_focus.initialize(
            config.width,
            config.height))
    {
        std::cerr
            << "Failed to initialize CUDA "
               "focus processor.\n";

        return EXIT_FAILURE;
    }


    // --------------------------------------------------------
    // GSTREAMER
    // --------------------------------------------------------

    gst_init(
        &argc,
        &argv);


    // ========================================================
    // CAMERA PIPELINE
    // ========================================================
    //
    // Camera
    //    |
    //    v
    // NV12 NVMM
    //    |
    //    v
    // nvvidconv
    //    |
    //    v
    // Y42B
    //    |
    //    v
    //   tee
    //   |
    // appsink
    //
    // If display is enabled:
    //
    //   tee
    //    |
    //   queue
    //    |
    // videoconvert
    //    |
    // autovideosink
    //
    // The display branch is added ONLY when --display is
    // enabled.
    // ========================================================

    std::string pipeline_string =
        "nvarguscamerasrc "
        "sensor-id=" +
        std::to_string(config.camera) +
        " ! "
        "video/x-raw(memory:NVMM),"
        "width=" +
        std::to_string(config.width) +
        ",height=" +
        std::to_string(config.height) +
        ",framerate=" +
        std::to_string(config.fps) +
        "/1,"
        "format=NV12 "
        "! "
        "nvvidconv "
        "! "
        "video/x-raw,format=Y42B "
        "! "
        "tee name=t "

        // ----------------------------------------------------
        // AUTOFOCUS / APPLICATION BRANCH
        // ----------------------------------------------------

        "t. ! "
        "queue "
        "max-size-buffers=4 "
        "max-size-bytes=0 "
        "max-size-time=0 "
        "leaky=downstream "
        "! "
        "appsink "
        "name=sink "
        "sync=false "
        "emit-signals=false "
        "max-buffers=4 "
        "drop=true ";

    // --------------------------------------------------------
    // OPTIONAL DISPLAY BRANCH
    // --------------------------------------------------------

    if (config.display)
    {
        pipeline_string +=
            "t. ! "
            "queue "
            "max-size-buffers=2 "
            "max-size-bytes=0 "
            "max-size-time=0 "
            "leaky=downstream "
            "! "
            "videoconvert "
            "! "
            "autovideosink "
            "sync=false ";
    }


    std::cout
        << "GStreamer pipeline:\n"
        << pipeline_string
        << "\n\n";


    GError* error = nullptr;

    GstElement* pipeline =
        gst_parse_launch(
            pipeline_string.c_str(),
            &error);

    if (!pipeline)
    {
        std::cerr
            << "Failed to create GStreamer pipeline.\n";

        if (error)
        {
            std::cerr
                << error->message
                << '\n';

            g_error_free(error);
        }

        gst_deinit();

        return EXIT_FAILURE;
    }

    if (error)
    {
        std::cerr
            << "GStreamer parse warning/error: "
            << error->message
            << '\n';

        g_error_free(error);

        gst_object_unref(pipeline);
        gst_deinit();

        return EXIT_FAILURE;
    }


    // --------------------------------------------------------
    // APPSINK
    // --------------------------------------------------------

    GstElement* sink_element =
        gst_bin_get_by_name(
            GST_BIN(pipeline),
            "sink");

    if (!sink_element)
    {
        std::cerr
            << "Unable to find appsink.\n";

        gst_object_unref(pipeline);
        gst_deinit();

        return EXIT_FAILURE;
    }

    GstAppSink* appsink =
        GST_APP_SINK(sink_element);

    g_object_set(
        G_OBJECT(appsink),
        "sync", FALSE,
        "emit-signals", FALSE,
        "max-buffers", 4,
        "drop", TRUE,
        nullptr);


    // --------------------------------------------------------
    // FFMPEG WRITER
    // --------------------------------------------------------
    //
    // It is only started if recording OR streaming is enabled.
    //
    // If both are disabled, no FFmpeg process is created.
    // Autofocus continues to operate from appsink.
    // --------------------------------------------------------

    FfmpegWriter writer(
        config.output,
        config.width,
        config.height,
        config.fps,
        config.alsa_device,
        config.record,
        config.stream,
        8);

    if (writer.enabled())
    {
        if (!writer.start())
        {
            gst_object_unref(sink_element);
            gst_object_unref(pipeline);
            gst_deinit();

            return EXIT_FAILURE;
        }
    }


    // --------------------------------------------------------
    // STATUS
    // --------------------------------------------------------

    std::cout
        << "Camera:       "
        << config.camera
        << '\n'
        << "Resolution:   "
        << config.width
        << " x "
        << config.height
        << '\n'
        << "FPS:          "
        << config.fps
        << '\n'
        << "Format:       Y42B / YUV422P"
        << '\n'
        << "Display:      "
        << (config.display
                ? "ENABLED"
                : "DISABLED")
        << '\n'
        << "Recording:    "
        << (config.record
                ? "ENABLED"
                : "DISABLED")
        << '\n'
        << "Streaming:    "
        << (config.stream
                ? "ENABLED"
                : "DISABLED")
        << '\n';

    if (config.record)
    {
        std::cout
            << "Output:       "
            << config.output
            << '\n';
    }

    if (config.stream)
    {
        std::cout
            << "VLC stream:   udp://@: "
            << VLC_STREAM_PORT
            << " (MPEG-TS)"
            << '\n';
    }

    std::cout
        << "Audio:        "
        << config.alsa_device
        << '\n'
        << "Autofocus:    CUDA hybrid metric / coarse-to-fine search"
        << '\n'
        << "\nPress Ctrl+C to stop.\n";


    // --------------------------------------------------------
    // START CAMERA
    // --------------------------------------------------------

    const GstStateChangeReturn state_result =
        gst_element_set_state(
            pipeline,
            GST_STATE_PLAYING);

    if (state_result ==
        GST_STATE_CHANGE_FAILURE)
    {
        std::cerr
            << "Failed to start GStreamer pipeline.\n";

        writer.stop();

        gst_element_set_state(
            pipeline,
            GST_STATE_NULL);

        gst_object_unref(sink_element);
        gst_object_unref(pipeline);
        gst_deinit();

        return EXIT_FAILURE;
    }


    // --------------------------------------------------------
    // AUTOFOCUS
    // --------------------------------------------------------

    AutofocusController autofocus;

    if (!autofocus.start(
            focus_controller))
    {
        std::cerr
            << "Failed to start autofocus.\n";

        gst_element_set_state(
            pipeline,
            GST_STATE_NULL);

        writer.stop();

        gst_object_unref(sink_element);
        gst_object_unref(pipeline);
        gst_deinit();

        return EXIT_FAILURE;
    }


    // --------------------------------------------------------
    // CAPTURE LOOP
    // --------------------------------------------------------

    guint64 frame_count = 0;

    while (g_running.load(
        std::memory_order_relaxed))
    {
        GstSample* sample =
            gst_app_sink_pull_sample(
                appsink);

        if (!sample)
        {
            if (g_running.load(
                    std::memory_order_relaxed))
            {
                std::cerr
                    << "appsink returned NULL.\n";
            }

            break;
        }


        GstVideoInfo info{};
        GstVideoFrame video_frame{};

        if (!get_video_frame(
                sample,
                info,
                video_frame))
        {
            std::cerr
                << "Invalid GStreamer video sample.\n";

            gst_sample_unref(sample);

            break;
        }


        const int width =
            GST_VIDEO_INFO_WIDTH(&info);

        const int height =
            GST_VIDEO_INFO_HEIGHT(&info);

        if (width != config.width ||
            height != config.height)
        {
            std::cerr
                << "Unexpected frame size: "
                << width
                << " x "
                << height
                << '\n';

            gst_video_frame_unmap(
                &video_frame);

            gst_sample_unref(sample);

            break;
        }


        // ----------------------------------------------------
        // YUV PLANES
        // ----------------------------------------------------

        const guint8* y =
            static_cast<const guint8*>(
                GST_VIDEO_FRAME_PLANE_DATA(
                    &video_frame,
                    0));

        const guint8* u =
            static_cast<const guint8*>(
                GST_VIDEO_FRAME_PLANE_DATA(
                    &video_frame,
                    1));

        const guint8* v =
            static_cast<const guint8*>(
                GST_VIDEO_FRAME_PLANE_DATA(
                    &video_frame,
                    2));


        const gint y_stride =
            GST_VIDEO_FRAME_PLANE_STRIDE(
                &video_frame,
                0);

        const gint u_stride =
            GST_VIDEO_FRAME_PLANE_STRIDE(
                &video_frame,
                1);

        const gint v_stride =
            GST_VIDEO_FRAME_PLANE_STRIDE(
                &video_frame,
                2);


        // ----------------------------------------------------
        // AUTOFOCUS
        // ----------------------------------------------------

        if (!autofocus.finished())
        {
            autofocus.process(
                focus_controller,
                cuda_focus,
                y,
                static_cast<size_t>(
                    y_stride));

            if (autofocus.failed())
            {
                g_running.store(
                    false,
                    std::memory_order_relaxed);
            }
        }


        // ----------------------------------------------------
        // COPY FRAME ONLY IF RECORDING OR STREAMING
        // ----------------------------------------------------
        //
        // When both are disabled, there is no reason to copy
        // the complete YUV frame into another buffer.
        //
        // This reduces CPU and memory bandwidth when running
        // autofocus-only mode.
        // --------------------------------------------------------

        if (writer.enabled())
        {
            const int chroma_width =
                width / 2;

            const size_t y_size =
                static_cast<size_t>(width) *
                static_cast<size_t>(height);

            const size_t chroma_size =
                static_cast<size_t>(chroma_width) *
                static_cast<size_t>(height);

            YuvFrame output_frame;

            output_frame.data.resize(
                y_size +
                2 * chroma_size);


            uint8_t* dst_y =
                output_frame.data.data();

            uint8_t* dst_u =
                dst_y + y_size;

            uint8_t* dst_v =
                dst_u + chroma_size;


            copy_plane(
                dst_y,
                y,
                y_stride,
                width,
                height);

            copy_plane(
                dst_u,
                u,
                u_stride,
                chroma_width,
                height);

            copy_plane(
                dst_v,
                v,
                v_stride,
                chroma_width,
                height);


            // ------------------------------------------------
            // UNMAP BEFORE QUEUING
            // ------------------------------------------------

            gst_video_frame_unmap(
                &video_frame);

            gst_sample_unref(
                sample);


            // ------------------------------------------------
            // QUEUE FOR FFMPEG
            // ------------------------------------------------

            if (!writer.push(
                    std::move(output_frame)))
            {
                if (writer.has_error())
                {
                    std::cerr
                        << "FFmpeg writer failed.\n";
                }
                else
                {
                    std::cerr
                        << "Unable to queue frame.\n";
                }

                g_running.store(
                    false,
                    std::memory_order_relaxed);

                break;
            }
        }
        else
        {
            // ------------------------------------------------
            // NO RECORDING AND NO STREAMING
            // ------------------------------------------------
            //
            // Only autofocus needed this frame.
            // No full-frame CPU copy is performed.
            // ------------------------------------------------

            gst_video_frame_unmap(
                &video_frame);

            gst_sample_unref(
                sample);
        }


        ++frame_count;


        // ----------------------------------------------------
        // STATUS
        // ----------------------------------------------------

        if (frame_count %
            static_cast<guint64>(
                config.fps) == 0)
        {
            if (writer.enabled())
            {
                std::cout
                    << "\rFrames processed: "
                    << frame_count
                    << "  Queue: "
                    << writer.queue_size()
                    << "    "
                    << std::flush;
            }
            else
            {
                std::cout
                    << "\rFrames processed: "
                    << frame_count
                    << "    "
                    << std::flush;
            }
        }
    }


    // --------------------------------------------------------
    // SHUTDOWN
    // --------------------------------------------------------

    std::cout
        << "\nStopping camera...\n";


    gst_element_set_state(
        pipeline,
        GST_STATE_NULL);


    writer.stop();


    gst_object_unref(
        sink_element);

    gst_object_unref(
        pipeline);

    gst_deinit();


    // --------------------------------------------------------
    // FINAL STATUS
    // --------------------------------------------------------

    std::cout
        << "Processed "
        << frame_count
        << " frames.\n";

    if (config.record)
    {
        std::cout
            << "Recorded video: "
            << config.output
            << '\n';
    }

    if (config.stream)
    {
        std::cout
            << "VLC stream: udp://@: "
            << VLC_STREAM_PORT
            << '\n';
    }


    if (writer.has_error())
    {
        std::cerr
            << "Recording/streaming failed.\n";

        return EXIT_FAILURE;
    }


    return EXIT_SUCCESS;
}