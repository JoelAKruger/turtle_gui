#include <stdint.h>
#include <stdio.h>
#include <assert.h>

#include <queue>
#include <thread>
#include <mutex>
#include <array>

#include <algorithm>
#include <numeric>
#include <cmath>

//Imgui
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

//FFMPEG
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
#include <libavutil/imgutils.h>
}

//OpenCV
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

//PulseAudio
#include <pulse/simple.h>
#include <pulse/error.h>

//STB image
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

GLuint load_texture(const char* filename) {
    int width = 0, height = 0, nrChannels = 0;
    unsigned char* data = stbi_load(filename, &width, &height, &nrChannels, 0);
    if (!data) {
        printf("Failed to load image: %s", filename);
        return 0;
    }

    GLenum format;
    if (nrChannels == 1)
        format = GL_RED;
    else if (nrChannels == 3)
        format = GL_RGB;
    else if (nrChannels == 4)
        format = GL_RGBA;
    else {
        stbi_image_free(data);
        printf("Unsupported channel count: %d", nrChannels);
        return 0;
    }

    GLuint textureID;
    glGenTextures(1, &textureID);
    glBindTexture(GL_TEXTURE_2D, textureID);

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, format,
                 GL_UNSIGNED_BYTE, data);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    stbi_image_free(data);

    return textureID;
}

std::array<const char*, 80> class_names = {
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "brocolli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush"
};


struct inference_detection
{
    float Confidence; 
    float X, Y, W, H; 
    int ClassID;
};

static std::vector<inference_detection>
RunInference(cv::dnn::Net& Network, cv::Size ModelShape, const cv::Mat& Frame, int ClassCount, float Threshold)
{
    cv::Mat Blob;
    cv::dnn::blobFromImage(Frame, Blob, 1.0f / 255.0f, ModelShape, cv::Scalar(), true, false);
    
    Network.setInput(Blob);
    
    std::vector<cv::Mat> Outputs;
    Network.forward(Outputs, Network.getUnconnectedOutLayersNames());
    
    int Rows = Outputs[0].size[2];
    int Dimensions = Outputs[0].size[1];
    
    float XScale = (float)Frame.cols / ModelShape.width;
    float YScale = (float)Frame.rows / ModelShape.height;
    
    Outputs[0] = Outputs[0].reshape(1, Dimensions);
    cv::transpose(Outputs[0], Outputs[0]);
    float* Data = (float*)Outputs[0].data;
    
    std::vector<inference_detection> Detections;
    
    std::vector<float> Confidences;
    std::vector<cv::Rect> Boxes;
    
    for (int Row = 0; Row < Rows; Row++)
    {
        float* ClassesScores = Data + 4;
        cv::Mat Scores(1, ClassCount, CV_32FC1, ClassesScores);
        cv::Point ClassID;
        double MaxClassScore;

        cv::minMaxLoc(Scores, 0, &MaxClassScore, 0, &ClassID);
        
        if (MaxClassScore > Threshold)
        {
            float X = Data[0] * XScale;
            float Y = Data[1] * YScale;
            float W = Data[2] * XScale;
            float H = Data[3] * YScale;
            
            inference_detection Inference = {
                (float)MaxClassScore, X, Y, W, H, ClassID.x
            };
            Detections.push_back(Inference);
            
            Boxes.emplace_back((int)(X - 0.5f * W), (int)(Y - 0.5f * H), (int)W, (int)H);
            Confidences.push_back((float)MaxClassScore);
        }
        
        Data += Dimensions;
    }
    
    float NMSThreshold = 0.4f;
    
    //Run non-maximum suppression to eliminate duplicate detections
    std::vector<int> NMSResult;
    cv::dnn::NMSBoxes(Boxes, Confidences, Threshold, NMSThreshold, NMSResult);
    
    std::vector<inference_detection> Result;
    
    for (int Index : NMSResult)
    {
        Result.push_back(Detections[Index]);
    }
    
    return Result;
}

struct CameraFeed
{
    const char* url = 0;
    std::queue<AVFrame*> frame_queue;
    std::mutex frame_queue_mutex;
};

void camera_receive_thread(CameraFeed* camera) 
{
    avformat_network_init();
    AVFormatContext* format_context = 0;
    AVDictionary *options = NULL;
    av_dict_set(&options, "probesize", "50000000", 0);       // 50 MB
    av_dict_set(&options, "analyzeduration", "10000000", 0); // 10s (in microseconds)
    av_dict_set(&options, "protocol_whitelist", "file,udp,rtp", 0);

    //avformat_open_input(&format_context, camera->url, 0, &options);
    avformat_open_input(&format_context, "stream.sdp", NULL, &options);
    avformat_find_stream_info(format_context, 0);

    int video_stream = -1;
    for (unsigned i = 0; i < format_context->nb_streams; i++) {
        if (format_context->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            video_stream = i;
        }
    }
    assert(video_stream >= 0);

    const AVCodec* codec = avcodec_find_decoder(format_context->streams[video_stream]->codecpar->codec_id);
    AVCodecContext* codec_context = avcodec_alloc_context3(codec);
    avcodec_parameters_to_context(codec_context, format_context->streams[video_stream]->codecpar);
    avcodec_open2(codec_context, codec, 0);

    AVPacket* packet = av_packet_alloc();
    AVFrame* frame = av_frame_alloc();
    AVFrame* rgb = av_frame_alloc();

    SwsContext* sws_context = sws_getContext(codec_context->width, codec_context->height, codec_context->pix_fmt,
                                         codec_context->width, codec_context->height, AV_PIX_FMT_RGB24,
                                         SWS_BILINEAR, 0, 0, 0);
    int byte_count = av_image_get_buffer_size(AV_PIX_FMT_RGB24, codec_context->width, codec_context->height, 1);
    std::vector<uint8_t> buffer(byte_count);
    av_image_fill_arrays(rgb->data, rgb->linesize, buffer.data(),
                         AV_PIX_FMT_RGB24, codec_context->width, codec_context->height, 1);

    while (av_read_frame(format_context, packet) >= 0) {
        if (packet->stream_index == video_stream) {
            if (avcodec_send_packet(codec_context, packet) == 0) {
                while (avcodec_receive_frame(codec_context, frame) == 0) {
                    // Allocate a fresh RGB frame
                    AVFrame* rgb = av_frame_alloc();
                    rgb->format = AV_PIX_FMT_RGB24;
                    rgb->width  = codec_context->width;
                    rgb->height = codec_context->height;

                    // Allocate buffers for it
                    if (av_frame_get_buffer(rgb, 1) < 0) {
                        fprintf(stderr, "Failed to allocate RGB frame buffer\n");
                        av_frame_free(&rgb);
                        continue;
                    }

                    // Convert YUV → RGB directly into this new frame
                    sws_scale(sws_context,
                            frame->data, frame->linesize,
                            0, codec_context->height,
                            rgb->data, rgb->linesize);

                    // Put it into your queue
                    {
                        std::lock_guard<std::mutex> lock(camera->frame_queue_mutex);
                        camera->frame_queue.push(rgb);
                    }
                }
            }
        }
        av_packet_unref(packet);
    }

    av_frame_free(&frame);
    av_frame_free(&rgb);
    av_packet_free(&packet);
    avcodec_free_context(&codec_context);
    avformat_close_input(&format_context);

    printf("Stream stopped\n");
}

struct AudioFeed
{
    const char* source = "rtp://0.0.0.0:4444";
    std::vector<float> waveform;
    std::vector<float> spectrum;
    std::queue<std::vector<int16_t>> playback_queue;
    std::mutex audio_mutex;
    std::mutex playback_mutex;
    int sample_rate = 8000;  // Match your FFmpeg streaming rate
    static constexpr int buffer_size = 8000;  // 1 second of audio at 8kHz
    static constexpr int fft_size = 512;
    int write_pos = 0;
    bool enable_playback = true;
    
    // update initial params in constructor using initial params
    AudioFeed() 
    {
        waveform.resize(buffer_size, 0.0f);
        spectrum.resize(fft_size / 2, 0.0f);
    }
};

void audio_receive_thread(AudioFeed* audio)
{
    avformat_network_init();
    AVFormatContext* format_context = 0;
    AVDictionary *options = NULL;
    av_dict_set(&options, "probesize", "50000000", 0);       // 50 MB
    av_dict_set(&options, "analyzeduration", "10000000", 0); // 10s (in microseconds)
    av_dict_set(&options, "protocol_whitelist", "file,udp,rtp", 0);

    //avformat_open_input(&format_context, camera->url, 0, &options);
    avformat_open_input(&format_context, "rtp://0.0.0.0:4444", NULL, &options);
    avformat_find_stream_info(format_context, 0);

    int audio_stream = -1;
    for (unsigned i = 0; i < format_context->nb_streams; i++) {
        if (format_context->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_AUDIO) {
            audio_stream = i;
        }
    }
    assert(audio_stream >= 0);

    const AVCodec* codec = avcodec_find_decoder(format_context->streams[audio_stream]->codecpar->codec_id);
    AVCodecContext* codec_context = avcodec_alloc_context3(codec);
    avcodec_parameters_to_context(codec_context, format_context->streams[audio_stream]->codecpar);
    avcodec_open2(codec_context, codec, 0);

    AVPacket* packet = av_packet_alloc();
    AVFrame* frame = av_frame_alloc();

    printf("Audio stream opened: %d Hz, %d channels\n", codec_context->sample_rate, codec_context->channels);

    while (av_read_frame(format_context, packet) >= 0) {
        if (packet->stream_index == audio_stream) {
            if (avcodec_send_packet(codec_context, packet) == 0) {
                while (avcodec_receive_frame(codec_context, frame) == 0) {
                    // Process audio frame
                    int samples_per_channel = frame->nb_samples;
                    std::vector<int16_t> playback_samples;
                    
                    for (int i = 0; i < samples_per_channel; i++) {
                        float sample = 0.0f;
                        int16_t playback_sample = 0;

                        // Handle different audio formats
                        if (frame->format == AV_SAMPLE_FMT_S16P) {
                            int16_t* data = (int16_t*)frame->data[0];
                            playback_sample = data[i];
                            sample = data[i] / 32768.0f;
                        }
                        
                        playback_samples.push_back(playback_sample);
                        
                        // Store in visualization buffer
                        {
                            std::lock_guard<std::mutex> lock(audio->audio_mutex);
                            audio->waveform[audio->write_pos] = sample;
                            audio->write_pos = (audio->write_pos + 1) % AudioFeed::buffer_size;
                        }
                    }
                    
                    // Add to playback queue
                    {
                        std::lock_guard<std::mutex> lock(audio->playback_mutex);
                        audio->playback_queue.push(playback_samples);
                        
                        // Limit queue size to prevent excessive buffering
                        while (audio->playback_queue.size() > 10) {
                            audio->playback_queue.pop();
                        }
                    }
                    
                    // Calculate spectrum periodically
                    static int spectrum_counter = 0;
                    spectrum_counter += samples_per_channel;
                    if (spectrum_counter >= AudioFeed::fft_size) {
                        spectrum_counter = 0;
                        
                        std::lock_guard<std::mutex> lock(audio->audio_mutex);
                        std::vector<float> fft_input(AudioFeed::fft_size);
                        for (int i = 0; i < AudioFeed::fft_size; i++) {
                            int pos = (audio->write_pos - AudioFeed::fft_size + i + AudioFeed::buffer_size) % AudioFeed::buffer_size;
                            fft_input[i] = audio->waveform[pos];
                        }
                        
                        // simple fft
                        int n = fft_input.size();
                        if (n <= 1) return;
                        
                        // Simple magnitude spectrum calculation using DFT
                        audio->spectrum.resize(n / 2);
                        for (int k = 0; k < n / 2; k++) {
                            float real = 0, imag = 0;
                            for (int i = 0; i < n; i++) {
                                float angle = -2.0f * M_PI * k * i / n;
                                real += fft_input[i] * cos(angle);
                                imag += fft_input[i] * sin(angle);
                            }
                            audio->spectrum[k] = sqrt(real * real + imag * imag);
                        }
                    }
                }
            }
        }
        av_packet_unref(packet);
    }
    
    // Cleanup
    av_frame_free(&frame);
    av_packet_free(&packet);
    avcodec_free_context(&codec_context);
    avformat_close_input(&format_context);
    
    printf("Audio stream stopped\n");
}

void audio_playback_thread(AudioFeed* audio) 
{
    // Set up PulseAudio for playback
    int const sample_rate = 8000;

    pa_sample_spec sample_spec = {
        .format = PA_SAMPLE_S16LE,
        .rate = sample_rate,  // Match your stream rate
        .channels = 1  // Mono
    };
    
    int error = 0;
    pa_simple* conn = pa_simple_new(
        0,           // Server
        "TurtleGUI",      // Application name
        PA_STREAM_PLAYBACK, // Direction
        audio->source,          // Device
        "Audio Playback", // Stream description
        &sample_spec,     // Sample format
        0,          // Channel map
        0,          // Buffering attributes
        &error            // Error code
    );
    assert(conn);

    printf("Audio playback thread started\n");
    
    while (true) {
        std::vector<int16_t> audio_chunk;
        
        // Get audio data from queue
        {
            std::lock_guard<std::mutex> lock(audio->playback_mutex);
            if (!audio->playback_queue.empty() && audio->enable_playback) {
                audio_chunk = audio->playback_queue.front();
                audio->playback_queue.pop();
            }
        }
        
        if (!audio_chunk.empty()) {
            // Play audio
            if (pa_simple_write(conn, audio_chunk.data(), 
                               audio_chunk.size() * sizeof(int16_t), &error) < 0) {
                printf("PulseAudio write error!");
            }
        } else {
            // Sleep briefly if no audio to play
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
    
    pa_simple_free(conn);
}

class Turtle
{
    //Cameras
    CameraFeed camera_feed;
    GLuint camera_texture, model_output_texture;
    GLuint drive_texture;
    GLuint drive_overlay_texture;

    //Audio
    AudioFeed audio_feed;

    //Yolo model
    char model_path[256];
    cv::dnn::Net onnx_network;
    std::string onnx_result;
    cv::Size onnx_model_shape = {640, 640};
    bool should_run_model = false;
    float model_confidence_threshold = 0.5f;

public:
    Turtle() {
        camera_feed.url = "udp://0.0.0.0:1234";
        glGenTextures(1, &camera_texture);
        glGenTextures(1, &model_output_texture);

        audio_feed.source = "rtp-recv.monitor";

        drive_texture = load_texture("src/drive.jpg");
        drive_overlay_texture = load_texture("src/drive_overlay.png");

        model_path[0] = 0;
    }

    void start_camera_thread() {
        std::thread camera(camera_receive_thread, &camera_feed);
        camera.detach();
    }

    void start_audio_thread() {
        std::thread audio(audio_receive_thread, &audio_feed);
        audio.detach();
    }

    void start_audio_playback_thread() {
        std::thread playback(audio_playback_thread, &audio_feed);
        playback.detach();
    }

    void receive_frames() {
        AVFrame* latest_frame = nullptr;
        {
            std::lock_guard<std::mutex> lock(camera_feed.frame_queue_mutex);
            while (!camera_feed.frame_queue.empty()) {
                if (latest_frame)
                    av_frame_free(&latest_frame);
                latest_frame = camera_feed.frame_queue.front();
                camera_feed.frame_queue.pop();
            }
        }

        if (latest_frame) {
            glBindTexture(GL_TEXTURE_2D, camera_texture);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, latest_frame->width, latest_frame->height, 0,
                        GL_RGB, GL_UNSIGNED_BYTE, latest_frame->data[0]);

            if (should_run_model) {
                cv::Mat frame(
                    latest_frame->height,
                    latest_frame->width,
                    CV_8UC3,
                    latest_frame->data[0],
                    latest_frame->linesize[0]
                );

                frame = frame.clone();

                std::vector<inference_detection> inferences;
                
                try {
                    inferences = RunInference(onnx_network, onnx_model_shape, frame, class_names.size(), model_confidence_threshold);
                }
                catch (cv::Exception exception) {
                }

                for (inference_detection& inference : inferences) {
                    cv::Rect rect = cv::Rect((int)(inference.X - 0.5f * inference.W), (int)(inference.Y - 0.5f * inference.H), (int)inference.W, (int)inference.H);
                    cv::rectangle(frame, rect, cv::Scalar(255, 255, 255), 3);

                    char label_buf[100];
                    snprintf(label_buf, sizeof(label_buf), "%s %.2f",
                            class_names[inference.ClassID],
                            inference.Confidence);

                    std::string label(label_buf);

                    cv::putText(frame, label,
                                cv::Point(rect.x, rect.y + 15), 
                                cv::FONT_HERSHEY_SIMPLEX,
                                0.5,
                                cv::Scalar(255, 255, 255),
                                    2);

                    printf("Detection\n");
                }

                // Upload to OpenGL
                glBindTexture(GL_TEXTURE_2D, model_output_texture);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, frame.cols, frame.rows, 0, GL_RGB, GL_UNSIGNED_BYTE, frame.data);

                should_run_model = false;
            }

            av_frame_free(&latest_frame);
        }
    }

    void display_camera() {
        ImGui::Begin("Camera");
        if (camera_texture) {
            constexpr float aspect_ratio = 16.0f / 9.0f;

            ImVec2 size = ImGui::GetContentRegionAvail();
            ImVec2 draw_size;

            if ((size.x / size.y) > aspect_ratio) {
                draw_size.y = size.y;
                draw_size.x = draw_size.y * aspect_ratio;
            }
            else {
                draw_size.x = size.x;
                draw_size.y = draw_size.x / aspect_ratio;
            }

            ImVec2 cursor = ImGui::GetCursorPos();
            ImGui::SetCursorPos(ImVec2(
                cursor.x + (size.x - draw_size.x) * 0.5f,
                cursor.y + (size.y - draw_size.y) * 0.5f
            ));

            ImGui::Image((ImTextureID)(intptr_t)camera_texture, draw_size);
        }
        ImGui::End();
    }

    void display_model() {
        ImGui::Begin("Model");
        ImGui::Text("Model path: ");
        ImGui::InputText("", model_path, sizeof(model_path));
        ImGui::SameLine();
        if (ImGui::Button("Open")) {
            try {
                onnx_network = cv::dnn::readNetFromONNX(model_path);
                onnx_result = std::string("Using model: ") + model_path;
            }
            catch (cv::Exception exception) {
                onnx_result = exception.err;
            }

        }

        ImGui::Text("%s", onnx_result.c_str());
        ImGui::SliderFloat("##xx", &model_confidence_threshold, 0.0f, 1.0f, "Confidence Threshold: %.3f");
        ImGui::SameLine();
        if (ImGui::Button("Run Model")) {
            should_run_model = true;
        }

        ImGui::Text("Model Output:");
        if (model_output_texture) {
            constexpr float aspect_ratio = 16.0f / 9.0f;

            ImVec2 size = ImGui::GetContentRegionAvail();
            ImVec2 draw_size;

            if ((size.x / size.y) > aspect_ratio) {
                draw_size.y = size.y;
                draw_size.x = draw_size.y * aspect_ratio;
            }
            else {
                draw_size.x = size.x;
                draw_size.y = draw_size.x / aspect_ratio;
            }

            ImVec2 cursor = ImGui::GetCursorPos();
            ImGui::SetCursorPos(ImVec2(
                cursor.x + (size.x - draw_size.x) * 0.5f,
                cursor.y + (size.y - draw_size.y) * 0.5f
            ));

            ImGui::Image((ImTextureID)(intptr_t)model_output_texture, draw_size);
        }

        ImGui::End();
    }

    void display_drive() {
        float velocity = 0.5f + 0.5f * sinf(ImGui::GetTime());
        float max_velocity = 1.0f;

        ImGui::Begin("Drive");
        ImGui::Text("Speed");
        ImGui::SameLine();
        char buf[64];
        sprintf(buf, "%.2f m/s", velocity);
        ImGui::ProgressBar(velocity, ImVec2(0.f, 0.f), buf);

        ImVec2 pos = ImGui::GetCursorScreenPos();
        ImVec2 size = ImVec2(600, 600);

        //Base
        ImGui::GetWindowDrawList()->AddImage((ImTextureID)(intptr_t)drive_texture, pos, ImVec2(pos.x + size.x, pos.y + size.y));

        ImGui::GetWindowDrawList()->AddImageQuad((ImTextureID)(intptr_t)drive_texture, ImVec2(100, 200), ImVec2(200, 100), ImVec2(100, 0), ImVec2(0, 100), ImVec2(0.0f, 1.0f), ImVec2(1.0f, 1.0f), ImVec2(1.0f, 0.0f), ImVec2(0.0f, 0.0f), 0xFFFFFFFF);


        //Overlay
        ImGui::GetWindowDrawList()->AddImage((ImTextureID)drive_overlay_texture, pos, ImVec2(pos.x + size.x, pos.y + size.y),
            ImVec2(0, 0), ImVec2(1, 1), IM_COL32(255, 255, 255, 255 * (velocity / max_velocity)));

        ImGui::Dummy(size); // reserve the layout space so ImGui knows something is here

        ImGui::End();
    }

    void display_audio() {
        ImGui::Begin("Audio");
        
        // Waveform display
        ImGui::Text("Waveform");
        
        std::vector<float> display_waveform;
        float rms_level = 0.0f;
        
        {
            std::lock_guard<std::mutex> lock(audio_feed.audio_mutex);
            
            // Get recent waveform data for display (last 1000 samples)
            int display_samples = std::min(1000, (int)audio_feed.waveform.size());
            display_waveform.resize(display_samples);
            
            for (int i = 0; i < display_samples; i++) {
                int pos = (audio_feed.write_pos - display_samples + i + AudioFeed::buffer_size) % AudioFeed::buffer_size;
                display_waveform[i] = audio_feed.waveform[pos];
                rms_level += display_waveform[i] * display_waveform[i];
            }
            
            rms_level = sqrt(rms_level / display_samples);
        }
        
        // Draw waveform
        if (!display_waveform.empty()) {
            ImGui::PlotLines("##waveform", display_waveform.data(), display_waveform.size(), 
                            0, nullptr, -1.0f, 1.0f, ImVec2(0, 100));
        }
        
        // Audio level meter
        ImGui::Text("Audio Level: %.3f", rms_level);
        ImGui::ProgressBar(std::min(rms_level * 10.0f, 1.0f), ImVec2(-1, 0), "");
        
        // Spectrum display
        ImGui::Text("Spectrum");
        
        std::vector<float> display_spectrum;
        {
            std::lock_guard<std::mutex> lock(audio_feed.audio_mutex);
            display_spectrum = audio_feed.spectrum;
        }
        
        if (!display_spectrum.empty()) {
            // Normalize spectrum for display
            float max_val = *std::max_element(display_spectrum.begin(), display_spectrum.end());
            if (max_val > 0) {
                for (float& val : display_spectrum) {
                    val /= max_val;
                }
            }
            
            ImGui::PlotHistogram("##spectrum", display_spectrum.data(), display_spectrum.size(),
                            0, nullptr, 0.0f, 1.0f, ImVec2(0, 150));
        }
        
        // Audio info
        ImGui::Text("Sample Rate: %d Hz", audio_feed.sample_rate);
        ImGui::Text("Buffer Size: %d samples", AudioFeed::buffer_size);

        // Playback controls
        ImGui::Checkbox("Enable Audio Playback", &audio_feed.enable_playback);
        
        // Queue status
        int queue_size = 0;
        {
            std::lock_guard<std::mutex> lock(audio_feed.playback_mutex);
            queue_size = audio_feed.playback_queue.size();
        }
        ImGui::Text("Playback Queue: %d chunks", queue_size);
        
        ImGui::End();
    }
    
    void render() {
        receive_frames();
        display_camera();
        display_model();
        display_drive();
        display_audio();
    }
};

int main()
{
    int glfw_init_result = glfwInit();
    assert(glfw_init_result);

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(1280, 720, "Turtle", NULL, NULL);
    assert(window);
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();

    ImGui::StyleColorsDark();

    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330"); 

    Turtle turtle;
    turtle.start_camera_thread();
    turtle.start_audio_thread();
    turtle.start_audio_playback_thread();

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        // Start ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        ImGuiStyle& style = ImGui::GetStyle();
        style.TabRounding = 8.f;
        style.FrameRounding = 8.f;
        style.GrabRounding = 8.f;
        style.WindowRounding = 8.f;
        style.PopupRounding = 8.f;

        turtle.render();

        // Render
        ImGui::Render();
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(0.2f, 0.2f, 0.2f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
    }

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}