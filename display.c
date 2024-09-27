#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <vector>

using namespace cv;
using namespace std;

// Function to draw a star on the background image
void draw_star(int x, int y, float magnitude, Mat* background, int ROI = 5) {
    double H = 90000 * exp(-magnitude + 1);
    double sigma = 0.5;

    for (int u = x - ROI; u <= x + ROI; u++) {
        for (int v = y - ROI; v <= y + ROI; v++) {
            double dist = (u - x) * (u - x) + (v - y) * (v - y);
            double diff = dist / (2 * (sigma * sigma));
            double exponent_exp = 1 / exp(diff);
            int raw_intensity = static_cast<int>(round((H / (2 * CV_PI * (sigma * sigma))) * exponent_exp));

            // Boundary check and set pixel intensity
            if (u >= 0 && u < background->cols && v >= 0 && v < background->rows) {
                (*background).at<uchar>(v, u) = static_cast<uchar>(min(raw_intensity, 255));
            }
        }
    }
}

int main() {
    // Load star data from CSV
    ifstream file("transfer_stars.csv");
    string line;
    vector<vector<float>> stars;

    // Parse CSV file
    while (getline(file, line)) {
        stringstream ss(line);
        float x, y, mag;
        ss >> x;
        ss.ignore(1, ',');
        ss >> y;
        ss.ignore(1, ',');
        ss >> mag;
        stars.push_back({x, y, mag});
    }

    // Create a background image
    Mat background = Mat::zeros(1400, 1400, CV_8UC1);

    // Draw stars on the background
    for (const auto& star : stars) {
        int x = static_cast<int>(star[0]);
        int y = static_cast<int>(star[1]);
        float mag = star[2];
        draw_star(x, y, mag, &background);
    }

    // Resize and save the output image
    Mat rescaled_image;
    resize(background, rescaled_image, Size(1400, 1400));

    // Save the image
    imwrite("star_image_csv_c.png", rescaled_image);

    return 0;
}
