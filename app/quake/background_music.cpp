#include "app/quake/background_music.hpp"

#define OV_EXCLUDE_STATIC_CALLBACKS
#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <print>
#include <utility>
#include <vorbis/vorbisfile.h>

namespace ds_vk_quake
{
using namespace ds_vk;

BackgroundMusic::BackgroundMusic(BackgroundMusic&& other) noexcept
    : spec_{other.spec_}, pcm_{std::move(other.pcm_)},
      stream_{std::exchange(other.stream_, nullptr)}, cursor_{std::exchange(other.cursor_, 0zu)},
      owns_audio_subsystem_{std::exchange(other.owns_audio_subsystem_, false)}
{
}

auto BackgroundMusic::operator=(BackgroundMusic&& other) noexcept -> BackgroundMusic&
{
    if (this == &other)
    {
        return *this;
    }
    reset();
    spec_ = other.spec_;
    pcm_ = std::move(other.pcm_);
    stream_ = std::exchange(other.stream_, nullptr);
    cursor_ = std::exchange(other.cursor_, 0zu);
    owns_audio_subsystem_ = std::exchange(other.owns_audio_subsystem_, false);
    return *this;
}

BackgroundMusic::~BackgroundMusic()
{
    reset();
}

auto BackgroundMusic::load_looped_ogg(const std::filesystem::path& filepath, f32 gain)
    -> std::optional<BackgroundMusic>
{
    const auto should_own_audio_subsystem = !SDL_WasInit(SDL_INIT_AUDIO);
    if (should_own_audio_subsystem && !SDL_InitSubSystem(SDL_INIT_AUDIO))
    {
        std::println(stderr, "Failed to initialize SDL audio: {}", SDL_GetError());
        return std::nullopt;
    }

    struct AudioSubsystemGuard
    {
        bool owns{};
        ~AudioSubsystemGuard()
        {
            if (owns)
            {
                SDL_QuitSubSystem(SDL_INIT_AUDIO);
            }
        }

        [[nodiscard]] auto release() -> bool
        {
            return std::exchange(owns, false);
        }
    } audio_guard{.owns = should_own_audio_subsystem};

    const auto filepath_string = filepath.string();
    OggVorbis_File vorbis{};
    if (const auto open_result = ov_fopen(filepath_string.c_str(), &vorbis); open_result != 0)
    {
        std::println(
            stderr,
            "Failed to open OGG music {}: ov_fopen returned {}",
            filepath_string,
            open_result
        );
        return std::nullopt;
    }

    struct VorbisGuard
    {
        OggVorbis_File* file{};
        ~VorbisGuard()
        {
            if (file != nullptr)
            {
                ov_clear(file);
            }
        }
    } vorbis_guard{.file = &vorbis};

    const auto* info = ov_info(&vorbis, -1);
    if (info == nullptr || info->channels <= 0 || info->rate <= 0
        || info->rate > std::numeric_limits<int>::max())
    {
        std::println(stderr, "Invalid Vorbis stream info in {}", filepath_string);
        return std::nullopt;
    }

    std::vector<std::byte> pcm{};
    if (const auto frame_count = ov_pcm_total(&vorbis, -1); frame_count > 0)
    {
        constexpr auto bytes_per_sample = 2;
        const auto bytes_per_frame = static_cast<ogg_int64_t>(info->channels) * bytes_per_sample;
        const auto byte_count = frame_count * bytes_per_frame;
        if (byte_count > 0
            && byte_count <= static_cast<ogg_int64_t>(std::numeric_limits<usize>::max()))
        {
            pcm.reserve(static_cast<usize>(byte_count));
        }
    }

    std::array<char, 32 * 1024> decode_buf{};
    auto bitstream = 0;
    while (true)
    {
        const auto bytes_read = ov_read(
            &vorbis, decode_buf.data(), static_cast<int>(decode_buf.size()), 0, 2, 1, &bitstream
        );
        if (bytes_read == 0)
        {
            break;
        }
        if (bytes_read < 0)
        {
            std::println(
                stderr,
                "Failed while decoding OGG music {}: ov_read returned {}",
                filepath_string,
                bytes_read
            );
            return std::nullopt;
        }

        const auto old_size = pcm.size();
        pcm.resize(old_size + static_cast<usize>(bytes_read));
        std::memcpy(pcm.data() + old_size, decode_buf.data(), static_cast<usize>(bytes_read));
    }

    if (pcm.empty())
    {
        std::println(stderr, "Decoded no PCM data from {}", filepath_string);
        return std::nullopt;
    }

    BackgroundMusic out{};
    out.spec_ = SDL_AudioSpec{
        .format = SDL_AUDIO_S16LE,
        .channels = info->channels,
        .freq = static_cast<int>(info->rate),
    };
    out.pcm_ = std::move(pcm);
    out.owns_audio_subsystem_ = audio_guard.release();
    out.stream_ =
        SDL_OpenAudioDeviceStream(SDL_AUDIO_DEVICE_DEFAULT_PLAYBACK, &out.spec_, nullptr, nullptr);
    if (out.stream_ == nullptr)
    {
        std::println(stderr, "Failed to open SDL audio stream: {}", SDL_GetError());
        return std::nullopt;
    }
    if (!SDL_SetAudioStreamGain(out.stream_, gain))
    {
        std::println(stderr, "Failed to set music gain: {}", SDL_GetError());
    }
    if (!SDL_ResumeAudioStreamDevice(out.stream_))
    {
        std::println(stderr, "Failed to start SDL audio stream: {}", SDL_GetError());
        return std::nullopt;
    }
    out.pump();
    std::println("Looping background music: {}", filepath.filename().string());
    return out;
}

auto BackgroundMusic::pump() -> void
{
    if (stream_ == nullptr || pcm_.empty())
    {
        return;
    }

    constexpr auto target_queued_bytes = 256 * 1024;
    constexpr auto max_chunk_bytes = 64zu * 1024zu;

    auto queued_bytes = SDL_GetAudioStreamQueued(stream_);
    if (queued_bytes < 0)
    {
        std::println(stderr, "Failed to query SDL audio stream queue: {}", SDL_GetError());
        return;
    }

    while (queued_bytes < target_queued_bytes)
    {
        const auto bytes_available = pcm_.size() - cursor_;
        const auto chunk_size = std::min(max_chunk_bytes, bytes_available);
        if (chunk_size == 0)
        {
            cursor_ = 0;
            continue;
        }

        const auto chunk_size_int = static_cast<int>(chunk_size);
        if (!SDL_PutAudioStreamData(stream_, pcm_.data() + cursor_, chunk_size_int))
        {
            std::println(stderr, "Failed to queue background music: {}", SDL_GetError());
            return;
        }
        cursor_ = (cursor_ + chunk_size) % pcm_.size();
        queued_bytes += chunk_size_int;
    }
}

auto BackgroundMusic::reset() -> void
{
    if (stream_ != nullptr)
    {
        SDL_DestroyAudioStream(stream_);
        stream_ = nullptr;
    }
    if (owns_audio_subsystem_)
    {
        SDL_QuitSubSystem(SDL_INIT_AUDIO);
        owns_audio_subsystem_ = false;
    }
}
}  // namespace ds_vk_quake
