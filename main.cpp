#include <GL/glut.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <ctime>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <direct.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace
{
constexpr int kWindowWidth = 960;
constexpr int kWindowHeight = 720;
constexpr int kWorldWidth = 240;
constexpr int kWorldHeight = 180;

constexpr float kGroundHeight = 18.0f;
constexpr float kBirdX = 58.0f;
constexpr float kBirdRadius = 6.0f;
constexpr float kGravity = -0.34f;
constexpr float kFlapVelocity = 4.9f;
constexpr float kPipeWidth = 22.0f;
constexpr float kStartPipeGap = 96.0f;
constexpr float kMinPipeGap = 38.0f;
constexpr float kStartPipeSpeed = 0.38f;
constexpr float kMaxPipeSpeed = 1.36f;
constexpr int kPipeCount = 4;
constexpr float kStartPipeSpacing = 104.0f;
constexpr float kMinPipeSpacing = 70.0f;
constexpr float kPipeStartX = 228.0f;
constexpr float kFixedDt = 1.0f / 60.0f;
constexpr int kDefaultSimulationSteps = 1;
constexpr int kRespawnDelayFrames = 18;

constexpr int kDyBins = 18;
constexpr int kVelBins = 12;
constexpr int kDistBins = 14;
constexpr int kActionCount = 2;
constexpr int kStateCount = kDyBins * kVelBins * kDistBins;

const std::string kTrainingDir = "training";
const std::string kQTableFile = "training/qtable.csv";
const std::string kStatsFile = "training/stats.txt";
const std::string kEpisodesFile = "training/episode_log.csv";

struct Color
{
    float r;
    float g;
    float b;
};

struct Vec2
{
    float x;
    float y;
};

struct Matrix3
{
    float m[3][3]{};
};

struct Pipe
{
    float x = 0.0f;
    float gapY = 90.0f;
    bool passed = false;
};

struct Bird
{
    float y = 96.0f;
    float velocity = 0.0f;
};

struct TrainingStats
{
    int episodes = 0;
    int bestScore = 0;
    long long totalSteps = 0;
    long long totalUpdates = 0;
    double epsilon = 0.18;
    double alpha = 0.16;
    double gamma = 0.92;
};

struct EpisodeSnapshot
{
    double reward = 0.0;
    int steps = 0;
};

std::mt19937 gRng(static_cast<unsigned int>(
    std::chrono::high_resolution_clock::now().time_since_epoch().count()));

Bird gBird;
std::vector<Pipe> gPipes;
std::vector<std::array<double, kActionCount>> gQTable(kStateCount, {0.0, 0.0});
TrainingStats gStats;
EpisodeSnapshot gEpisode;

bool gAlive = true;
bool gManualMode = false;
bool gPendingManualFlap = false;
bool gNeedsSaveNotice = false;
int gRespawnCounter = 0;
int gScore = 0;
int gSimulationSteps = kDefaultSimulationSteps;
int gLastChosenAction = 0;
double gLastReward = 0.0;
double gElapsedSeconds = 0.0;
std::chrono::steady_clock::time_point gLastSaveTime = std::chrono::steady_clock::now();

Matrix3 identityMatrix()
{
    Matrix3 out{};
    out.m[0][0] = 1.0f;
    out.m[1][1] = 1.0f;
    out.m[2][2] = 1.0f;
    return out;
}

Matrix3 translationMatrix(float tx, float ty)
{
    Matrix3 out = identityMatrix();
    out.m[0][2] = tx;
    out.m[1][2] = ty;
    return out;
}

Matrix3 rotationMatrix(float radians)
{
    Matrix3 out = identityMatrix();
    const float c = std::cos(radians);
    const float s = std::sin(radians);
    out.m[0][0] = c;
    out.m[0][1] = -s;
    out.m[1][0] = s;
    out.m[1][1] = c;
    return out;
}

Matrix3 scaleMatrix(float sx, float sy)
{
    Matrix3 out = identityMatrix();
    out.m[0][0] = sx;
    out.m[1][1] = sy;
    return out;
}

Matrix3 multiply(const Matrix3& a, const Matrix3& b)
{
    Matrix3 out{};
    for (int row = 0; row < 3; ++row)
    {
        for (int col = 0; col < 3; ++col)
        {
            out.m[row][col] = a.m[row][0] * b.m[0][col]
                            + a.m[row][1] * b.m[1][col]
                            + a.m[row][2] * b.m[2][col];
        }
    }
    return out;
}

Vec2 transformPoint(const Matrix3& matrix, const Vec2& point)
{
    return {
        matrix.m[0][0] * point.x + matrix.m[0][1] * point.y + matrix.m[0][2],
        matrix.m[1][0] * point.x + matrix.m[1][1] * point.y + matrix.m[1][2]
    };
}

int clampInt(int value, int low, int high)
{
    return std::max(low, std::min(value, high));
}

float clampFloat(float value, float low, float high)
{
    return std::max(low, std::min(value, high));
}

int roundToInt(float value)
{
    return static_cast<int>(std::lround(value));
}

float lerp(float start, float end, float t)
{
    return start + (end - start) * t;
}

std::string timeStamp()
{
    const std::time_t now = std::time(nullptr);
    std::tm tmValue{};
#if defined(_WIN32)
    localtime_s(&tmValue, &now);
#else
    localtime_r(&now, &tmValue);
#endif
    std::ostringstream out;
    out << std::put_time(&tmValue, "%Y-%m-%d %H:%M:%S");
    return out.str();
}

void ensureTrainingDirectory()
{
    const int result = _mkdir(kTrainingDir.c_str());
    if (result == -1 && errno != EEXIST)
    {
        std::cerr << "Failed to create training directory: " << kTrainingDir << "\n";
    }
}

float randomGapY()
{
    const float halfGap = lerp(kStartPipeGap, kMinPipeGap, 0.0f) * 0.5f;
    std::uniform_real_distribution<float> dist(kGroundHeight + halfGap + 10.0f,
                                               kWorldHeight - halfGap - 12.0f);
    return dist(gRng);
}

float difficultyProgress()
{
    const float elapsedSeconds = static_cast<float>(gEpisode.steps) * kFixedDt;
    const float timeProgress = clampFloat(elapsedSeconds / 150.0f, 0.0f, 1.0f);
    const float scoreProgress = clampFloat(static_cast<float>(gScore) / 38.0f, 0.0f, 1.0f);
    const float linearProgress = clampFloat(timeProgress * 0.60f + scoreProgress * 0.40f, 0.0f, 1.0f);
    return linearProgress * linearProgress * linearProgress * linearProgress;
}

float currentPipeGap()
{
    return lerp(kStartPipeGap, kMinPipeGap, difficultyProgress());
}

float currentPipeSpeed()
{
    return lerp(kStartPipeSpeed, kMaxPipeSpeed, difficultyProgress());
}

float currentPipeSpacing()
{
    return lerp(kStartPipeSpacing, kMinPipeSpacing, difficultyProgress());
}

float randomGapYForCurrentDifficulty()
{
    const float halfGap = currentPipeGap() * 0.5f;
    std::uniform_real_distribution<float> dist(kGroundHeight + halfGap + 10.0f,
                                               kWorldHeight - halfGap - 12.0f);
    return dist(gRng);
}

void drawFilledRect(float x1, float y1, float x2, float y2, const Color& color)
{
    glColor3f(color.r, color.g, color.b);
    glBegin(GL_QUADS);
    glVertex2f(x1, y1);
    glVertex2f(x2, y1);
    glVertex2f(x2, y2);
    glVertex2f(x1, y2);
    glEnd();
}

void putPixel(int x, int y, const Color& color)
{
    if (x < 0 || x >= kWorldWidth || y < 0 || y >= kWorldHeight)
    {
        return;
    }

    glColor3f(color.r, color.g, color.b);
    glBegin(GL_QUADS);
    glVertex2f(static_cast<float>(x), static_cast<float>(y));
    glVertex2f(static_cast<float>(x + 1), static_cast<float>(y));
    glVertex2f(static_cast<float>(x + 1), static_cast<float>(y + 1));
    glVertex2f(static_cast<float>(x), static_cast<float>(y + 1));
    glEnd();
}

void drawLineDDA(int x1, int y1, int x2, int y2, const Color& color)
{
    const int dx = x2 - x1;
    const int dy = y2 - y1;
    const int steps = std::max(std::abs(dx), std::abs(dy));

    if (steps == 0)
    {
        putPixel(x1, y1, color);
        return;
    }

    const float xIncrement = static_cast<float>(dx) / static_cast<float>(steps);
    const float yIncrement = static_cast<float>(dy) / static_cast<float>(steps);

    float x = static_cast<float>(x1);
    float y = static_cast<float>(y1);

    for (int i = 0; i <= steps; ++i)
    {
        putPixel(roundToInt(x), roundToInt(y), color);
        x += xIncrement;
        y += yIncrement;
    }
}

void drawLineBresenham(int x1, int y1, int x2, int y2, const Color& color)
{
    int x = x1;
    int y = y1;
    const int dx = std::abs(x2 - x1);
    const int dy = std::abs(y2 - y1);
    const int sx = (x1 < x2) ? 1 : -1;
    const int sy = (y1 < y2) ? 1 : -1;
    int err = dx - dy;

    while (true)
    {
        putPixel(x, y, color);
        if (x == x2 && y == y2)
        {
            break;
        }

        const int e2 = err * 2;
        if (e2 > -dy)
        {
            err -= dy;
            x += sx;
        }
        if (e2 < dx)
        {
            err += dx;
            y += sy;
        }
    }
}

void drawCircleSymmetryPoints(int cx, int cy, int x, int y, const Color& color)
{
    putPixel(cx + x, cy + y, color);
    putPixel(cx - x, cy + y, color);
    putPixel(cx + x, cy - y, color);
    putPixel(cx - x, cy - y, color);
    putPixel(cx + y, cy + x, color);
    putPixel(cx - y, cy + x, color);
    putPixel(cx + y, cy - x, color);
    putPixel(cx - y, cy - x, color);
}

void drawCircleHorizontalSpans(int cx, int cy, int x, int y, const Color& color)
{
    for (int i = cx - x; i <= cx + x; ++i)
    {
        putPixel(i, cy + y, color);
        putPixel(i, cy - y, color);
    }
    for (int i = cx - y; i <= cx + y; ++i)
    {
        putPixel(i, cy + x, color);
        putPixel(i, cy - x, color);
    }
}

void drawCircleMidpointOutline(int cx, int cy, int radius, const Color& color)
{
    int x = 0;
    int y = radius;
    int decision = 1 - radius;

    while (x <= y)
    {
        drawCircleSymmetryPoints(cx, cy, x, y, color);
        ++x;
        if (decision < 0)
        {
            decision += 2 * x + 1;
        }
        else
        {
            --y;
            decision += 2 * (x - y) + 1;
        }
    }
}

void fillCircleMidpoint(int cx, int cy, int radius, const Color& color)
{
    int x = 0;
    int y = radius;
    int decision = 1 - radius;

    while (x <= y)
    {
        drawCircleHorizontalSpans(cx, cy, x, y, color);
        ++x;
        if (decision < 0)
        {
            decision += 2 * x + 1;
        }
        else
        {
            --y;
            decision += 2 * (x - y) + 1;
        }
    }
}

void drawPolygonOutline(const std::vector<Vec2>& points, const Color& color, bool useDda)
{
    if (points.empty())
    {
        return;
    }

    for (std::size_t i = 0; i < points.size(); ++i)
    {
        const Vec2& a = points[i];
        const Vec2& b = points[(i + 1) % points.size()];
        if (useDda)
        {
            drawLineDDA(roundToInt(a.x), roundToInt(a.y), roundToInt(b.x), roundToInt(b.y), color);
        }
        else
        {
            drawLineBresenham(roundToInt(a.x), roundToInt(a.y), roundToInt(b.x), roundToInt(b.y), color);
        }
    }
}

void fillTriangle(Vec2 a, Vec2 b, Vec2 c, const Color& color)
{
    if (a.y > b.y)
    {
        std::swap(a, b);
    }
    if (a.y > c.y)
    {
        std::swap(a, c);
    }
    if (b.y > c.y)
    {
        std::swap(b, c);
    }

    auto interpolateX = [](const Vec2& from, const Vec2& to, float y) -> float
    {
        if (std::fabs(to.y - from.y) < 0.0001f)
        {
            return from.x;
        }
        const float ratio = (y - from.y) / (to.y - from.y);
        return from.x + ratio * (to.x - from.x);
    };

    const int yStart = roundToInt(a.y);
    const int yEnd = roundToInt(c.y);
    for (int y = yStart; y <= yEnd; ++y)
    {
        if (y < 0 || y >= kWorldHeight)
        {
            continue;
        }

        float xLeft = 0.0f;
        float xRight = 0.0f;

        if (y < b.y)
        {
            xLeft = interpolateX(a, b, static_cast<float>(y));
            xRight = interpolateX(a, c, static_cast<float>(y));
        }
        else
        {
            xLeft = interpolateX(b, c, static_cast<float>(y));
            xRight = interpolateX(a, c, static_cast<float>(y));
        }

        if (xLeft > xRight)
        {
            std::swap(xLeft, xRight);
        }

        for (int x = roundToInt(xLeft); x <= roundToInt(xRight); ++x)
        {
            putPixel(x, y, color);
        }
    }
}

void drawString(float x, float y, void* font, const std::string& text, const Color& color)
{
    glColor3f(color.r, color.g, color.b);
    glRasterPos2f(x, y);
    for (unsigned char character : text)
    {
        glutBitmapCharacter(font, character);
    }
}

void resetBird()
{
    gBird.y = 96.0f;
    gBird.velocity = 0.0f;
}

void initializePipes()
{
    gPipes.clear();
    const float pipeSpacing = currentPipeSpacing();
    for (int i = 0; i < kPipeCount; ++i)
    {
        Pipe pipe;
        pipe.x = kPipeStartX + static_cast<float>(i) * pipeSpacing;
        pipe.gapY = randomGapYForCurrentDifficulty();
        pipe.passed = false;
        gPipes.push_back(pipe);
    }
}

void resetRound()
{
    resetBird();
    initializePipes();
    gAlive = true;
    gPendingManualFlap = false;
    gRespawnCounter = 0;
    gScore = 0;
    gEpisode = {};
    gLastReward = 0.0;
    gLastChosenAction = 0;
}

int stateIndex(float birdY, float birdVelocity, const Pipe& nextPipe)
{
    const float dy = birdY - nextPipe.gapY;
    const float dx = (nextPipe.x + kPipeWidth) - kBirdX;

    const int dyBin = clampInt(static_cast<int>((dy + 90.0f) / 10.0f), 0, kDyBins - 1);
    const int velBin = clampInt(static_cast<int>((birdVelocity + 9.0f) / 1.5f), 0, kVelBins - 1);
    const int distBin = clampInt(static_cast<int>(dx / 10.0f), 0, kDistBins - 1);

    return (dyBin * kVelBins + velBin) * kDistBins + distBin;
}

const Pipe& nextPipeForBirdConst()
{
    for (const Pipe& pipe : gPipes)
    {
        if (pipe.x + kPipeWidth >= kBirdX - kBirdRadius)
        {
            return pipe;
        }
    }
    return gPipes.front();
}

int chooseAction(int state)
{
    std::uniform_real_distribution<double> chance(0.0, 1.0);
    if (chance(gRng) < gStats.epsilon)
    {
        std::uniform_int_distribution<int> randomAction(0, kActionCount - 1);
        return randomAction(gRng);
    }

    return (gQTable[state][1] > gQTable[state][0]) ? 1 : 0;
}

void updateQValue(int state, int action, double reward, int nextState, bool terminal)
{
    const double current = gQTable[state][action];
    const double nextBest = terminal
        ? 0.0
        : std::max(gQTable[nextState][0], gQTable[nextState][1]);
    gQTable[state][action] = current + gStats.alpha * (reward + gStats.gamma * nextBest - current);
    ++gStats.totalUpdates;
}

void appendEpisodeRecord()
{
    ensureTrainingDirectory();
    const bool newFile = !std::ifstream(kEpisodesFile).good();
    std::ofstream out(kEpisodesFile, std::ios::app);
    if (!out)
    {
        return;
    }

    if (newFile)
    {
        out << "timestamp,episode,score,best_score,epsilon,reward,steps\n";
    }

    out << timeStamp() << ','
        << gStats.episodes << ','
        << gScore << ','
        << gStats.bestScore << ','
        << std::fixed << std::setprecision(6) << gStats.epsilon << ','
        << std::fixed << std::setprecision(4) << gEpisode.reward << ','
        << gEpisode.steps << '\n';
}

void saveTrainingData()
{
    ensureTrainingDirectory();

    {
        std::ofstream out(kQTableFile);
        if (out)
        {
            out << "state,stay,flap\n";
            for (int state = 0; state < kStateCount; ++state)
            {
                out << state << ','
                    << std::fixed << std::setprecision(8) << gQTable[state][0] << ','
                    << std::fixed << std::setprecision(8) << gQTable[state][1] << '\n';
            }
        }
    }

    {
        std::ofstream out(kStatsFile);
        if (out)
        {
            out << "episodes=" << gStats.episodes << '\n';
            out << "best_score=" << gStats.bestScore << '\n';
            out << "total_steps=" << gStats.totalSteps << '\n';
            out << "total_updates=" << gStats.totalUpdates << '\n';
            out << std::fixed << std::setprecision(8);
            out << "epsilon=" << gStats.epsilon << '\n';
            out << "alpha=" << gStats.alpha << '\n';
            out << "gamma=" << gStats.gamma << '\n';
            out << "last_saved=" << timeStamp() << '\n';
        }
    }

    gLastSaveTime = std::chrono::steady_clock::now();
    gNeedsSaveNotice = true;
}

void loadStats()
{
    std::ifstream in(kStatsFile);
    if (!in)
    {
        return;
    }

    std::string line;
    while (std::getline(in, line))
    {
        const std::size_t split = line.find('=');
        if (split == std::string::npos)
        {
            continue;
        }

        const std::string key = line.substr(0, split);
        const std::string value = line.substr(split + 1);

        if (key == "episodes")
        {
            gStats.episodes = std::stoi(value);
        }
        else if (key == "best_score")
        {
            gStats.bestScore = std::stoi(value);
        }
        else if (key == "total_steps")
        {
            gStats.totalSteps = std::stoll(value);
        }
        else if (key == "total_updates")
        {
            gStats.totalUpdates = std::stoll(value);
        }
        else if (key == "epsilon")
        {
            gStats.epsilon = std::stod(value);
        }
        else if (key == "alpha")
        {
            gStats.alpha = std::stod(value);
        }
        else if (key == "gamma")
        {
            gStats.gamma = std::stod(value);
        }
    }
}

void loadQTable()
{
    std::ifstream in(kQTableFile);
    if (!in)
    {
        return;
    }

    std::string line;
    std::getline(in, line);
    while (std::getline(in, line))
    {
        std::stringstream row(line);
        std::string token;

        std::getline(row, token, ',');
        if (token.empty())
        {
            continue;
        }
        const int state = std::stoi(token);
        if (state < 0 || state >= kStateCount)
        {
            continue;
        }

        std::getline(row, token, ',');
        gQTable[state][0] = std::stod(token);
        std::getline(row, token, ',');
        gQTable[state][1] = std::stod(token);
    }
}

void loadTrainingData()
{
    ensureTrainingDirectory();
    loadStats();
    loadQTable();
}

void clearTrainingData()
{
    gQTable.assign(kStateCount, {0.0, 0.0});
    gStats = {};
    gStats.epsilon = 0.18;
    gStats.alpha = 0.16;
    gStats.gamma = 0.92;
    resetRound();
    saveTrainingData();

    std::ofstream out(kEpisodesFile);
    if (out)
    {
        out << "timestamp,episode,score,best_score,epsilon,reward,steps\n";
    }
}

void recyclePipe(Pipe& pipe, float newX)
{
    pipe.x = newX;
    pipe.gapY = randomGapYForCurrentDifficulty();
    pipe.passed = false;
}

bool birdHitsPipe(const Pipe& pipe)
{
    const float pipeGap = currentPipeGap();
    const float pipeLeft = pipe.x;
    const float pipeRight = pipe.x + kPipeWidth;
    const float gapBottom = pipe.gapY - pipeGap * 0.5f;
    const float gapTop = pipe.gapY + pipeGap * 0.5f;

    const bool intersectsX = (kBirdX + kBirdRadius > pipeLeft) && (kBirdX - kBirdRadius < pipeRight);
    const bool outsideGap = (gBird.y - kBirdRadius < gapBottom) || (gBird.y + kBirdRadius > gapTop);
    return intersectsX && outsideGap;
}

void finishEpisode()
{
    gAlive = false;
    gRespawnCounter = kRespawnDelayFrames;
    ++gStats.episodes;
    gStats.bestScore = std::max(gStats.bestScore, gScore);
    gStats.epsilon = std::max(0.02, gStats.epsilon * 0.9987);
    appendEpisodeRecord();
    saveTrainingData();
}

void finishManualRound()
{
    gAlive = false;
    gPendingManualFlap = false;
}

void stepSimulation()
{
    if (!gAlive)
    {
        if (gManualMode)
        {
            return;
        }

        if (--gRespawnCounter <= 0)
        {
            resetRound();
        }
        return;
    }

    int currentState = 0;
    int action = 0;
    if (gManualMode)
    {
        action = gPendingManualFlap ? 1 : 0;
    }
    else
    {
        const Pipe& currentPipe = nextPipeForBirdConst();
        currentState = stateIndex(gBird.y, gBird.velocity, currentPipe);
        action = chooseAction(currentState);
    }
    gLastChosenAction = action;
    gPendingManualFlap = false;

    if (action == 1)
    {
        gBird.velocity = kFlapVelocity;
    }

    gBird.velocity += kGravity;
    gBird.y += gBird.velocity;

    double reward = 0.08;
    bool collision = false;
    const float pipeSpeed = currentPipeSpeed();

    float furthestX = 0.0f;
    for (Pipe& pipe : gPipes)
    {
        pipe.x -= pipeSpeed;
        furthestX = std::max(furthestX, pipe.x);

        if (!pipe.passed && pipe.x + kPipeWidth < kBirdX)
        {
            pipe.passed = true;
            ++gScore;
            reward += 9.0;
        }

        if (birdHitsPipe(pipe))
        {
            collision = true;
        }
    }

    for (Pipe& pipe : gPipes)
    {
        if (pipe.x + kPipeWidth < -6.0f)
        {
            recyclePipe(pipe, furthestX + currentPipeSpacing());
            furthestX = pipe.x;
        }
    }

    if (gBird.y - kBirdRadius <= kGroundHeight || gBird.y + kBirdRadius >= kWorldHeight - 2.0f)
    {
        collision = true;
    }

    if (!gManualMode)
    {
        if (collision)
        {
            reward = -100.0;
        }
        else
        {
            const float gapOffset = std::fabs(gBird.y - nextPipeForBirdConst().gapY);
            reward += std::max(0.0, 1.6 - gapOffset * 0.02);
        }

        const int nextState = stateIndex(gBird.y, gBird.velocity, nextPipeForBirdConst());
        updateQValue(currentState, action, reward, nextState, collision);
        gEpisode.reward += reward;
        ++gStats.totalSteps;
    }
    else
    {
        reward = 0.0;
    }

    gLastReward = reward;
    ++gEpisode.steps;
    gElapsedSeconds += kFixedDt;

    if (collision)
    {
        if (gManualMode)
        {
            finishManualRound();
        }
        else
        {
            finishEpisode();
        }
    }

    const auto now = std::chrono::steady_clock::now();
    if (!gManualMode && now - gLastSaveTime > std::chrono::seconds(5))
    {
        saveTrainingData();
    }
}

void drawBackground()
{
    drawFilledRect(0.0f, 0.0f, static_cast<float>(kWorldWidth), static_cast<float>(kWorldHeight), {0.67f, 0.87f, 0.98f});
    drawFilledRect(0.0f, 0.0f, static_cast<float>(kWorldWidth), kGroundHeight, {0.74f, 0.58f, 0.29f});
    drawFilledRect(0.0f, kGroundHeight, static_cast<float>(kWorldWidth), kGroundHeight + 4.0f, {0.50f, 0.84f, 0.26f});

    for (int x = 0; x < kWorldWidth; x += 12)
    {
        drawLineDDA(x, kWorldHeight - 24, x + 20, kWorldHeight - 8, {0.82f, 0.94f, 1.0f});
    }

    for (int x = 0; x < kWorldWidth; x += 8)
    {
        drawLineBresenham(x, static_cast<int>(kGroundHeight), x + 5, static_cast<int>(kGroundHeight), {0.28f, 0.50f, 0.16f});
    }

    fillCircleMidpoint(202, 150, 11, {0.99f, 0.88f, 0.28f});
    drawCircleMidpointOutline(202, 150, 11, {0.94f, 0.61f, 0.11f});
}

void drawPipe(const Pipe& pipe)
{
    const float pipeGap = currentPipeGap();
    const float gapBottom = pipe.gapY - pipeGap * 0.5f;
    const float gapTop = pipe.gapY + pipeGap * 0.5f;
    const float left = pipe.x;
    const float right = pipe.x + kPipeWidth;

    drawFilledRect(left, kGroundHeight, right, gapBottom, {0.20f, 0.72f, 0.24f});
    drawFilledRect(left - 2.0f, gapBottom - 4.0f, right + 2.0f, gapBottom, {0.15f, 0.60f, 0.17f});
    drawFilledRect(left, gapTop, right, static_cast<float>(kWorldHeight), {0.20f, 0.72f, 0.24f});
    drawFilledRect(left - 2.0f, gapTop, right + 2.0f, gapTop + 4.0f, {0.15f, 0.60f, 0.17f});

    drawLineBresenham(roundToInt(left), roundToInt(kGroundHeight), roundToInt(left), roundToInt(gapBottom), {0.05f, 0.39f, 0.09f});
    drawLineBresenham(roundToInt(right), roundToInt(kGroundHeight), roundToInt(right), roundToInt(gapBottom), {0.05f, 0.39f, 0.09f});
    drawLineBresenham(roundToInt(left), roundToInt(gapTop), roundToInt(left), kWorldHeight - 1, {0.05f, 0.39f, 0.09f});
    drawLineBresenham(roundToInt(right), roundToInt(gapTop), roundToInt(right), kWorldHeight - 1, {0.05f, 0.39f, 0.09f});

    drawLineDDA(roundToInt(left + 4.0f), roundToInt(kGroundHeight + 4.0f), roundToInt(left + 4.0f), roundToInt(gapBottom - 5.0f), {0.56f, 0.92f, 0.56f});
    drawLineDDA(roundToInt(left + 4.0f), roundToInt(gapTop + 4.0f), roundToInt(left + 4.0f), kWorldHeight - 5, {0.56f, 0.92f, 0.56f});
}

void drawBird()
{
    const float pulse = 1.0f + 0.08f * std::sin(static_cast<float>(gElapsedSeconds) * 5.0f);
    const float wingTilt = std::clamp(-gBird.velocity * 0.06f, -0.8f, 0.9f);

    const Matrix3 translate = translationMatrix(kBirdX, gBird.y);
    const Matrix3 bodyScale = scaleMatrix(pulse, pulse);
    const Matrix3 bodyTransform = multiply(translate, bodyScale);
    const Matrix3 wingTransform = multiply(translate, multiply(rotationMatrix(wingTilt), scaleMatrix(1.0f, pulse)));
    const Matrix3 beakTransform = multiply(translate, rotationMatrix(-0.1f));

    const Vec2 bodyCenter = transformPoint(bodyTransform, {0.0f, 0.0f});
    const Vec2 eyeCenter = transformPoint(bodyTransform, {3.0f, 2.0f});
    const int bodyRadius = std::max(4, roundToInt(kBirdRadius * pulse));
    const int eyeRadius = std::max(1, roundToInt(2.2f * pulse));

    fillCircleMidpoint(roundToInt(bodyCenter.x), roundToInt(bodyCenter.y), bodyRadius, {0.99f, 0.91f, 0.20f});
    drawCircleMidpointOutline(roundToInt(bodyCenter.x), roundToInt(bodyCenter.y), bodyRadius, {0.85f, 0.56f, 0.08f});

    fillCircleMidpoint(roundToInt(eyeCenter.x), roundToInt(eyeCenter.y), eyeRadius, {1.0f, 1.0f, 1.0f});
    fillCircleMidpoint(roundToInt(eyeCenter.x + 1.0f), roundToInt(eyeCenter.y), 1, {0.12f, 0.12f, 0.12f});

    std::vector<Vec2> wing = {
        {-3.0f, 0.0f},
        {-11.0f, -3.0f},
        {-4.0f, -7.0f}
    };
    for (Vec2& point : wing)
    {
        point = transformPoint(wingTransform, point);
    }
    fillTriangle(wing[0], wing[1], wing[2], {0.96f, 0.72f, 0.12f});
    drawPolygonOutline(wing, {0.66f, 0.38f, 0.04f}, false);

    std::vector<Vec2> beak = {
        {6.0f, 0.0f},
        {12.0f, 2.0f},
        {6.0f, 4.0f}
    };
    for (Vec2& point : beak)
    {
        point = transformPoint(beakTransform, point);
    }
    fillTriangle(beak[0], beak[1], beak[2], {1.0f, 0.54f, 0.15f});
    drawPolygonOutline(beak, {0.72f, 0.28f, 0.08f}, true);

    drawLineBresenham(roundToInt(kBirdX - 2.0f), roundToInt(gBird.y - 7.0f),
                      roundToInt(kBirdX + 2.0f), roundToInt(gBird.y - 8.0f),
                      {0.74f, 0.44f, 0.08f});
}

void drawHud()
{
    drawFilledRect(4.0f, 154.0f, 110.0f, 176.0f, {0.05f, 0.16f, 0.22f});
    drawLineDDA(4, 165, 110, 165, {0.20f, 0.54f, 0.69f});

    std::ostringstream line1;
    line1 << "Score: " << gScore << "   Best: " << gStats.bestScore;
    drawString(8.0f, 168.0f, GLUT_BITMAP_8_BY_13, line1.str(), {1.0f, 1.0f, 1.0f});

    std::ostringstream line2;
    if (gManualMode)
    {
        line2 << "Mode: MANUAL  Control: SPACE";
    }
    else
    {
        line2 << "Episode: " << gStats.episodes
              << "  Eps: " << std::fixed << std::setprecision(3) << gStats.epsilon;
    }
    drawString(8.0f, 159.0f, GLUT_BITMAP_8_BY_13, line2.str(), {0.90f, 0.97f, 1.0f});

    std::ostringstream line3;
    if (gManualMode)
    {
        line3 << "Action: " << (gLastChosenAction == 1 ? "FLAP" : "GLIDE")
              << "  Diff: " << std::fixed << std::setprecision(0) << difficultyProgress() * 100.0f << "%";
    }
    else
    {
        line3 << "Action: " << (gLastChosenAction == 1 ? "FLAP" : "GLIDE")
              << "  R: " << std::fixed << std::setprecision(2) << gLastReward
              << "  Diff: " << std::fixed << std::setprecision(0) << difficultyProgress() * 100.0f << "%";
    }
    drawString(8.0f, 150.0f, GLUT_BITMAP_8_BY_13, line3.str(), {0.93f, 0.91f, 0.67f});

    drawString(8.0f, 7.0f, GLUT_BITMAP_8_BY_13,
               "SPACE: flap in manual mode | G: toggle RL/manual | RL autosaves to training/",
               {0.18f, 0.12f, 0.08f});

    if (!gAlive)
    {
        drawFilledRect(62.0f, 82.0f, 178.0f, 104.0f, {0.18f, 0.07f, 0.07f});
        if (gManualMode)
        {
            drawString(68.0f, 92.0f, GLUT_BITMAP_HELVETICA_18,
                       "Press SPACE to retry", {1.0f, 0.90f, 0.84f});
        }
        else
        {
            drawString(71.0f, 92.0f, GLUT_BITMAP_HELVETICA_18,
                       "Training next episode...", {1.0f, 0.90f, 0.84f});
        }
    }

    if (gNeedsSaveNotice)
    {
        drawString(148.0f, 168.0f, GLUT_BITMAP_8_BY_13,
                   "Checkpoint saved", {0.12f, 0.44f, 0.10f});
    }
}

void display()
{
    glClear(GL_COLOR_BUFFER_BIT);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    drawBackground();
    for (const Pipe& pipe : gPipes)
    {
        drawPipe(pipe);
    }
    drawBird();
    drawHud();

    glutSwapBuffers();
}

void timer(int)
{
    gNeedsSaveNotice = false;
    const int simulationSteps = gManualMode ? 1 : gSimulationSteps;
    for (int i = 0; i < simulationSteps; ++i)
    {
        stepSimulation();
    }
    glutPostRedisplay();
    glutTimerFunc(16, timer, 0);
}

void reshape(int width, int height)
{
    glViewport(0, 0, width, height);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0, static_cast<double>(kWorldWidth), 0.0, static_cast<double>(kWorldHeight), -1.0, 1.0);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}

void onKeyboard(unsigned char key, int, int)
{
    switch (key)
    {
    case 27:
        saveTrainingData();
        std::exit(0);
        break;
    case '+':
    case '=':
        gSimulationSteps = std::min(gSimulationSteps + 1, 16);
        break;
    case '-':
    case '_':
        gSimulationSteps = std::max(gSimulationSteps - 1, 1);
        break;
    case ' ':
        if (gManualMode)
        {
            if (!gAlive)
            {
                resetRound();
            }
            gPendingManualFlap = true;
        }
        break;
    case 'g':
    case 'G':
        gManualMode = !gManualMode;
        resetRound();
        break;
    case 'r':
    case 'R':
        clearTrainingData();
        break;
    default:
        break;
    }
}

void initializeOpenGL()
{
    glClearColor(0.67f, 0.87f, 0.98f, 1.0f);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_DITHER);
}

void onClose()
{
    saveTrainingData();
}

} // namespace

int main(int argc, char** argv)
{
    loadTrainingData();
    resetRound();
    std::atexit(onClose);

    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
    glutInitWindowSize(kWindowWidth, kWindowHeight);
    glutCreateWindow("Flappy Bird RL - GLUT/OpenGL");

    initializeOpenGL();

    glutDisplayFunc(display);
    glutReshapeFunc(reshape);
    glutKeyboardFunc(onKeyboard);
    glutTimerFunc(16, timer, 0);

    glutMainLoop();
    return 0;
}
