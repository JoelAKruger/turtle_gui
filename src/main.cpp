#include <stdio.h>
#include <assert.h>

#include <queue>
#include <thread>
#include <mutex>

#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
#include <libavutil/imgutils.h>
}

std::queue<AVFrame*> frame_queue;
std::mutex frame_queue_mutex;

void camera_receive_thread(const char* url) {
    avformat_network_init();
    AVFormatContext* format_context = 0;
    avformat_open_input(&format_context, url, 0, 0);
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
                        std::lock_guard<std::mutex> lock(frame_queue_mutex);
                        frame_queue.push(rgb);
                        printf("Got frame\n");
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
}

void run_gui()
{
    // UI
    ImGui::Begin("Hello, world!");
    ImGui::Text("This is Dear ImGui running on GLFW.");
    ImGui::End();
}

int main()
{
    int glfw_init_result = glfwInit();
    assert(glfw_init_result);

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(800, 600, "Dear ImGui + GLFW", NULL, NULL);
    assert(window);
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();

    ImGui::StyleColorsDark();

    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330");

    const char* url = "udp://0.0.0.0:1234";
    std::thread recv(camera_receive_thread, url);
    GLuint tex = 0;
    glGenTextures(1, &tex);

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        // --- Get the latest frame ---
        AVFrame* latest_frame = nullptr;
        {
            std::lock_guard<std::mutex> lock(frame_queue_mutex);
            while (!frame_queue.empty()) {
                if (latest_frame)
                    av_frame_free(&latest_frame);  // free old frames
                latest_frame = frame_queue.front();
                frame_queue.pop();
            }
        }

        if (latest_frame) {
            glBindTexture(GL_TEXTURE_2D, tex);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, latest_frame->width, latest_frame->height, 0,
                        GL_RGB, GL_UNSIGNED_BYTE, latest_frame->data[0]);
            av_frame_free(&latest_frame);
        }

        // Start ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // --- Display texture in ImGui ---
        ImGui::Begin("Camera");
        if (tex)
            ImGui::Image((ImTextureID)(intptr_t)tex, ImVec2(640, 480));
        ImGui::End();

        // Render
        ImGui::Render();
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
    }

    glDeleteTextures(1, &tex);

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}