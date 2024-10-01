#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <time.h>
#include <math.h>
#include <X11/Xlib.h>
#include <X11/Xutil.h>

#define PORT 8080
#define BUFFER_SIZE (12 * 1024)  //initialling calculated based on size of the CSV file
#define MAX_FLOATS (300 * 3)   //initialling calculated based on number of stars

Display *display;
Window window;
GC gc;

void init_xlib() {
    display = XOpenDisplay(NULL);
    if (!display) {
        fprintf(stderr, "Unable to open X display\n");
        exit(1);
    }
    int screen = DefaultScreen(display);    // specify which screen you need to connect if multiple screen exist
    window = XCreateSimpleWindow(display, RootWindow(display, screen), 0, 0, 1400, 1400, 1,
                                  BlackPixel(display, screen), BlackPixel(display, screen)); // Black background
    XSelectInput(display, window, ExposureMask | KeyPressMask);
    XMapWindow(display, window);
    gc = XCreateGC(display, window, 0, NULL);
}

void draw_star_xlib(Display* display, Window window, GC gc, int x, int y, float magnitude, int ROI) {
    double H = 90000 * exp(-magnitude + 1);
    double sigma = 0.5;
    
    for (int u = x - ROI; u <= x + ROI; u++) {
        for (int v = y - ROI; v <= y + ROI; v++) {
            double dist = (u - x) * (u - x) + (v - y) * (v - y);
            double diff = dist / (2 * (sigma * sigma));
            double exponent_exp = exp(-diff);
            int raw_intensity = (int)round((H / (2 * M_PI * (sigma * sigma))) * exponent_exp);


            if(ROI >= 5){
                // Ensure pixel coordinates are within bounds
                if (u >= 0 && u < 1400 && v >= 0 && v < 1400) {

                    int temp = fmin(raw_intensity, 255);
                    unsigned long gray_pixel = ((unsigned long)temp << 16) | 
                                                ((unsigned long)temp << 8) | 
                                                ((unsigned long)temp);

                    XSetForeground(display, gc, gray_pixel);
                    XDrawPoint(display, window, gc, u, v);

                }
            }
            else{
                if (u >= 0 && u < 1400 && v >= 0 && v < 1400) {
                    XSetForeground(display, gc, WhitePixel(display, DefaultScreen(display)));
                    XDrawPoint(display, window, gc, u, v);
                }
            }

        }
    }
}

void handle_client(int client_socket, int roi) {
    char buffer[BUFFER_SIZE];
    char full_data[BUFFER_SIZE] = {0};
    ssize_t bytes_received;

    init_xlib();

    const char *message = "READY\n";
    send(client_socket, message, strlen(message), 0);
    printf("Sent: %s", message);

    while (1) {

        clock_t t;
        t = clock();

        bytes_received = recv(client_socket, buffer, BUFFER_SIZE - 1, 0);
        if (bytes_received <= 0) {
            printf("Error or connection closed\n");
            break;
        }
        buffer[bytes_received] = '\0';

        printf("Received %ld bytes\n", bytes_received);

        // Set the background color to black and clear the window
        unsigned long background_color = BlackPixel(display, DefaultScreen(display));
        XSetWindowBackground(display, window, background_color);
        XClearWindow(display, window); // Clear with the black background


        if (buffer[0] == '$' && strchr(buffer, '&') != NULL) {           //check for star $ and end & delimeter
            // create a new char array full_data without $ and &
            char *end_ptr = strchr(buffer, '&');
            *end_ptr = '\0';
            strncat(full_data, buffer + 1, sizeof(full_data) - strlen(full_data) - 1);

            float float_array[MAX_FLOATS];
            size_t float_count = 0;

            for (size_t i = 0; i < strlen(full_data); i++) {            // reading buffer data and converting new_line on end of csv file to , 
                if (full_data[i] == '\n') {                             // so, we can tokenize the entire i/p data based on comma
                    full_data[i] = ',';
                }
            }

            char *token = strtok(full_data, ",");                       //get the first token
            while (token != NULL && float_count < MAX_FLOATS) {
                float_array[float_count++] = strtof(token, NULL);       //converting str to float and appending it in float_array
                token = strtok(NULL, ",");                              //update the token
            }

            for (size_t i = 0; i < float_count; i += 3) {               // iterate through the float_array
                if (i + 2 < float_count) {                              // and display it
                    int x = (int)float_array[i];
                    int y = (int)float_array[i + 1];
                    float magnitude = float_array[i + 2];
                    draw_star_xlib(display, window, gc, x, y, magnitude, roi);

                }
            }

            XFlush(display);                                            // Update the window
            // for(double long i=0; i<10000000; i++);

            const char *ack = "ACK: RD.\n";                             // send the ACK
            send(client_socket, ack, strlen(ack), 0);                   // RD read and Display
            printf("Sent: %s", ack);

        } else {
            const char *ack = "ACK: Waiting for proper data format.\n";
            send(client_socket, ack, strlen(ack), 0);
            printf("Sent: %s", ack);
        }

        memset(full_data, 0, sizeof(full_data));                        //reset full data for nxt iteration

        t = clock() - t; 
        double time_taken = ((double)t)/CLOCKS_PER_SEC;
        printf("Single execution took %f seconds to execute \n", time_taken);
    }

    XFreeGC(display, gc);                                              // free the graphics context
    XDestroyWindow(display, window);                                   // Destroy the window
    XCloseDisplay(display);                                            // close the X11 Display connection
}

int main() {
    int server_socket, client_socket;                                   // create necessary var for server, client add
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
    // server_addr.sin_addr.s_addr = inet_addr("SPECIFIC_IP");        // for running in specific IP
    // server_addr.sin_port = htons(SPECIFIC_PORT)roi;
    if (bind(server_socket, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
        perror("Bind failed");
        close(server_socket);
        exit(EXIT_FAILURE);
    }

    listen(server_socket, 3);                                       // 3 more connections can be queued
    printf("server is up on port %d...\n", PORT);


    FILE *file = fopen("config.txt", "r");
    if (file == NULL) {
        perror("Error opening configuration file");
        return EXIT_FAILURE;
    }

    char line[30];
    int roi = 0;

    while (fgets(line, sizeof(line), file)) {
        // Skip comments
        if (line[0] == '#') continue;

        // Parse ROI value
        if (sscanf(line, "ROI=%d", &roi) == 1) {
            break; 
        }
    }

    fclose(file);

    printf("The value of ROI is %d\n", roi);


    while (1) {
        client_socket = accept(server_socket, (struct sockaddr*)&client_addr, &client_addr_len);
        if (client_socket < 0) {
            perror("Accept failed ERROR :/");
            continue;
        }

        printf("Client connected :)\n");
        handle_client(client_socket, roi);
        close(client_socket);
    }

    close(server_socket);
    return 0;
}

