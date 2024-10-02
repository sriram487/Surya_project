#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <time.h>
#include <math.h>
#include <stdbool.h>
#include <X11/Xlib.h>
#include <X11/Xutil.h>

#define PORT 8080
#define BUFFER_SIZE (6 * 1024)
#define MAX_FLOATS (118 * 3)
#define MAX_PIXELS 7500

Display *display;
Window window;
GC gc;

// void init_xlib() {
//     display = XOpenDisplay(NULL);
//     if (!display) {
//         fprintf(stderr, "Unable to open X display\n");
//         exit(1);
//     }
//     int screen = DefaultScreen(display);
//     window = XCreateSimpleWindow(display, RootWindow(display, screen), 0, 0, 1400, 1400, 1,
//                                   BlackPixel(display, screen), BlackPixel(display, screen)); // Black background
//     XSelectInput(display, window, ExposureMask | KeyPressMask);
//     XMapWindow(display, window);
//     gc = XCreateGC(display, window, 0, NULL);
// }

// void draw_star_xlib(int x, int y, float magnitude, int ROI) {
//     double H = 90000 * exp(-magnitude + 1);
//     double sigma = 0.5;

//     for (int u = x - ROI; u <= x + ROI; u++) {
//         for (int v = y - ROI; v <= y + ROI; v++) {
//             double dist = (u - x) * (u - x) + (v - y) * (v - y);
//             double diff = dist / (2 * (sigma * sigma));
//             double exponent_exp = exp(-diff);
//             int raw_intensity = (int)round((H / (2 * M_PI * (sigma * sigma))) * exponent_exp);

//             if (u >= 0 && u < 1400 && v >= 0 && v < 1400) {
//                 // Draw point if intensity is above 0
//                 if (raw_intensity > 0) {
//                     XSetForeground(display, gc, WhitePixel(display, DefaultScreen(display)));
//                     XDrawPoint(display, window, gc, u, v);
//                 }
//             }
//         }
//     }
// }

void init_xlib(){
    
    display = XOpenDisplay(NULL);
    if (!display) {
        fprintf(stderr, "Unable to open X display\n");
        exit(1);
    }

    int screen = DefaultScreen(display);
    window = XCreateSimpleWindow(display, RootWindow(display, screen), 0, 0, 1400, 1400, 0,
                                    BlackPixel(display, screen), BlackPixel(display, screen));

    XSetWindowAttributes attrs;
    attrs.override_redirect = True; // Request the window manager to ignore this window's decorations
    XChangeWindowAttributes(display, window, CWOverrideRedirect, &attrs);

    XSelectInput(display, window, ExposureMask | KeyPressMask);
    XMapWindow(display, window);
    
    // Retrieve and print the window attributes
    XWindowAttributes attributes;
    XGetWindowAttributes(display, window, &attributes);

    int width = attributes.width;
    int height = attributes.height;

    printf("Window size: %d x %d\n", width, height);

    gc = XCreateGC(display, window, 0, NULL);

    // Initial draw to set the background
    XSetForeground(display, gc, BlackPixel(display, screen));
    XFillRectangle(display, window, gc, 0, 0, 1400, 1400);

}

void draw_star_xlib(Display *display, Window window, GC gc, XImage *image, int x, int y, float magnitude, int ROI, unsigned long* pixel_array, bool reset) {

    double H = 90000 * exp(-magnitude + 1);
    double sigma = 0.5;
    int pixel_index = 0;

    for (int u = x - ROI; u <= x + ROI; u++) {
        for (int v = y - ROI; v <= y + ROI; v++) {
            double dist = (u - x) * (u - x) + (v - y) * (v - y);
            double diff = dist / (2 * (sigma * sigma));
            double exponent_exp = exp(-diff);
            int raw_intensity = (int)round((H / (2 * M_PI * (sigma * sigma))) * exponent_exp);

            // Ensure pixel coordinates are within bounds
            if (u >= 0 && u < 1400 && v >= 0 && v < 1400) {

                // Get the pixel value at (u, v)
                unsigned long pixel_value = XGetPixel(image, u, v);

                // printf("%lu\n", pixel_value);

                // Store pixel values in pixel_array if there's space
                if (pixel_index < MAX_PIXELS) {
                    pixel_array[pixel_index] = pixel_value;
                    pixel_index++;
                }
                
                else {
                    fprintf(stderr, "Warning: Pixel value array is full. Some data may be lost.\n");
                }

                if(ROI >= 5){
                    
                    if (reset == false){

                        unsigned long temp = fmin(raw_intensity, 255);
                        
                        unsigned long gray_pixel = (temp << 16) | 
                                                    (temp << 8) | 
                                                    (temp);

                        unsigned long new_pixel = gray_pixel + pixel_value;

                        XSetForeground(display, gc, new_pixel);
                        XDrawPoint(display, window, gc, u, v);
                    }
                    else{

                        unsigned long temp = fmin(raw_intensity, 255);

                        unsigned long gray_pixel = (temp << 16) | 
                                                    (temp << 8) | 
                                                    (temp);

                        unsigned long new_pixel = gray_pixel - pixel_value;

                        XSetForeground(display, gc, new_pixel);
                        XDrawPoint(display, window, gc, u, v);

                    }
                }

                else{

                    if (reset == false){

                        XSetForeground(display, gc, WhitePixel(display, DefaultScreen(display)));
                        XDrawPoint(display, window, gc, u, v);
                    
                    }
                }
            }
        }
    }
    // memset(pixel_val, 0, sizeof(pixel_val));
}

void handle_client(int client_socket) {
    char buffer[BUFFER_SIZE];
    char full_data[BUFFER_SIZE] = {0};
    unsigned long pixel_array[MAX_PIXELS];
    ssize_t bytes_received;
    bool reset = false;

    init_xlib();

    while (1) {

        const char *message = "READY\n";
        send(client_socket, message, strlen(message), 0);
        printf("Sent: %s", message);

        bytes_received = recv(client_socket, buffer, BUFFER_SIZE - 1, 0);
        if (bytes_received <= 0) {
            printf("Error or connection closed\n");
            break;
        }
        buffer[bytes_received] = '\0';

        clock_t t;
        t = clock();

        printf("Received %ld bytes\n", bytes_received);

        // Set the background color to black and clear the window
        // unsigned long background_color = BlackPixel(display, DefaultScreen(display));
        // XSetWindowBackground(display, window, background_color);
        // XClearWindow(display, window); // Clear with the black background

        if (buffer[0] == '$' && strchr(buffer, '&') != NULL) {
            char *end_ptr = strchr(buffer, '&');
            *end_ptr = '\0';
            strncat(full_data, buffer + 1, sizeof(full_data) - strlen(full_data) - 1);

            float float_array[MAX_FLOATS];
            size_t float_count = 0;

            for (size_t i = 0; i < strlen(full_data); i++) {
                if (full_data[i] == '\n') {
                    full_data[i] = ',';
                }
            }

            char *token = strtok(full_data, ",");
            while (token != NULL && float_count < MAX_FLOATS) {
                float_array[float_count++] = strtof(token, NULL);
                token = strtok(NULL, ",");
            }

            
            // Capture the entire image of the window
            XImage *image = XGetImage(display, window, 0, 0, 1400, 1400, AllPlanes, ZPixmap);
            if (!image) {
                fprintf(stderr, "Failed to get image from window\n");
                return;
            }

            // displaying the stars
            for (size_t i = 0; i < float_count; i += 3) {
                if (i + 2 < float_count) {
                    int x = (int)float_array[i];
                    int y = (int)float_array[i + 1];
                    float magnitude = float_array[i + 2];
                    // draw_star_xlib(x, y, magnitude, 5);
                    draw_star_xlib(display, window, gc, image, x, y, magnitude, 5, pixel_array, reset);
                }
            } 

            // resetting the stars
            reset = true;
            if(reset == true){
                for (size_t i = 0; i < float_count; i += 3) {
                    if (i + 2 < float_count) {
                        int x = (int)float_array[i];
                        int y = (int)float_array[i + 1];
                        float magnitude = float_array[i + 2];
                        // draw_star_xlib(x, y, magnitude, 5);
                        draw_star_xlib(display, window, gc, image, x, y, magnitude, 5, pixel_array, reset);
                    }
                }
            }
            reset = false;
           
            const char *ack = "ACK: RD.\n";
            send(client_socket, ack, strlen(ack), 0);
            printf("Sent: %s", ack);

            XFlush(display); // Update the window

            // Free the image after use
            XDestroyImage(image);

        }

        else {
            const char *ack = "ACK: Waiting for proper data format.\n";
            send(client_socket, ack, strlen(ack), 0);
            printf("Sent: %s", ack);
        }

        memset(full_data, 0, sizeof(full_data));
        memset(pixel_array, 0, sizeof(pixel_array));

        t = clock() - t; 
        double time_taken = ((double)t)/CLOCKS_PER_SEC;
        printf("Single execution took %f seconds to execute \n", time_taken);
    }

    XFreeGC(display, gc);
    XDestroyWindow(display, window);
    XCloseDisplay(display);
}

int main() {
    int server_socket, client_socket;
    struct sockaddr_in server_addr, client_addr;
    socklen_t client_addr_len = sizeof(client_addr);

    server_socket = socket(AF_INET, SOCK_STREAM, 0);
    if (server_socket < 0) {
        perror("Socket creation failed");
        exit(EXIT_FAILURE);
    }

    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;
    server_addr.sin_port = htons(PORT);
    if (bind(server_socket, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
        perror("Bind failed");
        close(server_socket);
        exit(EXIT_FAILURE);
    }

    listen(server_socket, 3);
    printf("Server is listening on port %d...\n", PORT);

    while (1) {
        client_socket = accept(server_socket, (struct sockaddr*)&client_addr, &client_addr_len);
        if (client_socket < 0) {
            perror("Accept failed");
            continue;
        }

        printf("Client connected.\n");
        handle_client(client_socket);
        close(client_socket);
    }

    close(server_socket);
    return 0;
}

