#pragma once

#include <vector>
#include <array>
#include <queue>
#include <utility>

#include <cudaDefs.h>

#include "map.h"

using std::vector;
using std::array;
using std::queue;
using std::pair;

class FlowField {
public:
    FlowField(int gridSize)
        : gridSize(gridSize), flowField(gridSize, vector<array<uint8_t, 2>>(gridSize, { 255, 255 }))
    {
        directions = {
            { -1, 0 },  // Up
            { 1, 0 },   // Down
            { 0, -1 },  // Left
            { 0, 1 },   // Right
            { -1, -1 }, // Up-Left
            { -1, 1 },  // Up-Right
            { 1, -1 },  // Down-Left
            { 1, 1 }    // Down-Right
        };
    }

    inline int getSize()
    {
        return this->gridSize;
    }

    array<uint8_t, 2>& getData(int x, int y)
    {
        if (x >= this->gridSize || y >= this->gridSize)
        {
            static std::array<uint8_t, 2> empty = { 0, 0 };
            return empty;
        }

        return this->flowField[x][y];
    }


    void generateFlowFieldForMap(Map* map) {
        int goalX = map->goalPossition.x;
        int goalY = map->goalPossition.y;

        std::queue<std::pair<int, int>> q;
        std::vector<std::vector<bool>> visited(gridSize, std::vector<bool>(gridSize, false));
        q.push({ goalX, goalY });
        visited[goalX][goalY] = true;
        flowField[goalX][goalY] = { 128, 128 }; // Goal

        while (!q.empty()) {
            std::pair<int, int> current = q.front();
            q.pop();
            int cx = current.first;
            int cy = current.second;

            for (size_t i = 0; i < directions.size(); ++i) {
                int dx = directions[i].first;
                int dy = directions[i].second;
                int nx = cx + dx;
                int ny = cy + dy;

                if (nx >= 0 && ny >= 0 && nx < gridSize && ny < gridSize &&
                    !map->internalMap[nx][ny] && !visited[nx][ny]) {

                    int fromX = cx - nx;
                    int fromY = cy - ny;
                    std::array<uint8_t, 2> dir = encodeDirection(fromX, fromY);

                    flowField[nx][ny] = dir;
                    visited[nx][ny] = true;
                    q.push({ nx, ny });
                }
            }
        }
    }

    inline array<uint8_t, 2> encodeDirection(int dx, int dy) {
        if (dx == -1 && dy == 0)  return { 128, 255 }; // Up
        if (dx == 1 && dy == 0)   return { 128, 0 };   // Down
        if (dx == 0 && dy == -1)  return { 0, 128 };   // Left
        if (dx == 0 && dy == 1)   return { 255, 128 }; // Right
        if (dx == -1 && dy == -1) return { 0, 255 };   // Up-Left
        if (dx == -1 && dy == 1)  return { 255, 255 }; // Up-Right
        if (dx == 1 && dy == -1)  return { 0, 0 };     // Down-Left
        if (dx == 1 && dy == 1)   return { 255, 0 };   // Down-Right
        return { 128, 128 }; // No direction
    }

    inline void printFlowField() {
        for (int i = 0; i < gridSize; ++i) {
            for (int j = 0; j < gridSize; ++j) {
                uint8_t fx = flowField[i][j][0];
                uint8_t fy = flowField[i][j][1];

                if (fx == 128 && fy == 128) {
                    printf(" ");  // Goal
                }
                else if (fx == 128 && fy == 255) printf("^");   // Up
                else if (fx == 128 && fy == 0)   printf("v");   // Down
                else if (fx == 0 && fy == 128)   printf("<");   // Left
                else if (fx == 255 && fy == 128) printf(">");   // Right
                else if (fx == 0 && fy == 255)   printf("\\");  // Up-Left
                else if (fx == 255 && fy == 255) printf("/");   // Up-Right
                else if (fx == 0 && fy == 0)     printf("/");   // Down-Left
                else if (fx == 255 && fy == 0)   printf("\\");  // Down-Right
                else printf("?");  // Unknown
            }
            printf("\n");
        }
    }

    cudaArray* getRGBTexture(Map* map) {
        size_t width = gridSize;
        size_t height = gridSize;

        vector<uchar4> flatData;
        flatData.reserve(width * height);

        for (size_t y = 0; y < height; ++y) {
            for (size_t x = 0; x < width; ++x) {
                uchar4 pixel;
                pixel.x = flowField[y][x][0];         // R = flow X
                pixel.y = flowField[y][x][1];         // G = flow Y
                pixel.z = map->internalMap[y][x] ? 1 : 0; // B = 1 if wall, 0 otherwise
                pixel.w = 0;                           // A = unused

                flatData.push_back(pixel);
            }
        }

        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar4>();
        cudaArray* cuArray;
        cudaMallocArray(&cuArray, &channelDesc, width, height);

        cudaMemcpy2DToArray(
            cuArray,
            0, 0,
            flatData.data(),
            width * sizeof(uchar4),
            width * sizeof(uchar4),
            height,
            cudaMemcpyHostToDevice
        );

        return cuArray;
    }

private:
    int gridSize;
    vector<vector<array<uint8_t, 2>>> flowField;
    vector<pair<int, int>> directions;
};

