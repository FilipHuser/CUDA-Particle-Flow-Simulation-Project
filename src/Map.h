#pragma once

#include <vector_types.h>

class Map
{
public:
    Map(int gridSize)
    {
        this->internalMap = new bool* [gridSize];
        for (int i = 0; i < gridSize; ++i) {
            internalMap[i] = new bool[gridSize];
        }

        for (int i = 0; i < gridSize; ++i) {
            for (int j = 0; j < gridSize; ++j) {
                internalMap[i][j] = false; // False = open space, True = obstacle
            }
        }
        this->startPosition = { 0,0 };
        this->goalPossition = { 0,0 };
        this->gridSize = gridSize;
    }
    ~Map() noexcept
    {
        for (int i = 0; i < gridSize; ++i)
        {
            delete[] internalMap[i];
        }
        delete[] internalMap;
        internalMap = nullptr;
    }

    inline bool getCell(int x, int y)
    {
        if (x >= this->gridSize || y >= this->gridSize) return false;
        return this->internalMap[x][y];
    }


    inline void setWall(int x, int y)
    {
        if (x >= this->gridSize || y >= this->gridSize) return;
        this->internalMap[x][y] = true; // Mark as obstacle
    }

    inline void removeWall(int x, int y)
    {
        if (x >= this->gridSize || y >= this->gridSize) return;
        this->internalMap[x][y] = false; // Mark as open space
    }

    inline void setStart(int x, int y)
    {
        if (x >= this->gridSize || y >= this->gridSize) return;
        this->startPosition.x = x;
        this->startPosition.y = y;
    }

    inline void setGoal(int x, int y)
    {
        if (x >= this->gridSize || y >= this->gridSize) return;
        this->goalPossition.x = x;
        this->goalPossition.y = y;
    }

public:
    int2 startPosition;
    int2 goalPossition;
    bool** internalMap;
    int gridSize;
};

