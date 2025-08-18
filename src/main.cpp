#include <stdio.h>
#include <assert.h>

#include <queue>
#include <thread>
#include <mutex>
#include <array>

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

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

std::queue<AVFrame*> frame_queue;
std::mutex frame_queue_mutex;

void camera_receive_thread(const char* url) {
    avformat_network_init();
    AVFormatContext* format_context = 0;
    AVDictionary *options = NULL;
    av_dict_set(&options, "probesize", "50000000", 0);       // 50 MB
    av_dict_set(&options, "analyzeduration", "10000000", 0); // 10s (in microseconds)

    avformat_open_input(&format_context, url, 0, &options);
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
struct inference_detection
{
    float Confidence; 
    float X, Y, W, H; 
    int ClassID;
};

static std::vector<inference_detection>
RunInference(cv::dnn::Net& Network, cv::Size ModelShape, const cv::Mat& Frame, int ClassCount)
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
    
    float Threshold = 0.1f;
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

int main()
{
    int glfw_init_result = glfwInit();
    assert(glfw_init_result);

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(800, 600, "Turtle", NULL, NULL);
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
    GLuint camera_texture = 0;
    glGenTextures(1, &camera_texture);

    GLuint model_output_texture = 0;
    glGenTextures(1, &model_output_texture);

    char model_path[256] = {};
    
    cv::dnn::Net onnx_network;
    std::string onnx_result;
    cv::Size onnx_model_shape = {640, 640};

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

    bool should_run_model = false;

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        // --- Get the latest frame ---
        AVFrame* latest_frame = nullptr;
        {
            std::lock_guard<std::mutex> lock(frame_queue_mutex);
            while (!frame_queue.empty()) {
                if (latest_frame)
                    av_frame_free(&latest_frame);
                latest_frame = frame_queue.front();
                frame_queue.pop();
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

                std::vector<inference_detection> inferences = RunInference(onnx_network, onnx_model_shape, frame, class_names.size());

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

        // --- Display texture in ImGui ---
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