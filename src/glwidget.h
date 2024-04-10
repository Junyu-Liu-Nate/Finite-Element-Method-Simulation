#pragma once

#ifdef __APPLE__
#define GL_SILENCE_DEPRECATION
#endif

#include "simulation.h"
#include "graphics/camera.h"
#include "graphics/shader.h"

#include <QOpenGLWidget>
#include <QElapsedTimer>
#include <QTimer>
#include <memory>

class GLWidget : public QOpenGLWidget
{
    Q_OBJECT

public:
//    GLWidget(QWidget *parent = nullptr);
    explicit GLWidget(const Settings& settings, QWidget *parent = nullptr);
    ~GLWidget();

private:
    static const int FRAMES_TO_AVERAGE = 30;

private:
    // Basic OpenGL Overrides
    void initializeGL()         override;
    void paintGL()              override;
    void resizeGL(int w, int h) override;

    // Event Listeners
    void mousePressEvent  (QMouseEvent *event) override;
    void mouseMoveEvent   (QMouseEvent *event) override;
    void mouseReleaseEvent(QMouseEvent *event) override;
    void wheelEvent       (QWheelEvent *event) override;
    void keyPressEvent    (QKeyEvent   *event) override;
    void keyReleaseEvent  (QKeyEvent   *event) override;

private:
    QElapsedTimer m_deltaTimeProvider; // For measuring elapsed time
    QTimer        m_intervalTimer;     // For triggering timed events

    Simulation m_sim;
    Camera     m_camera;
    Shader    *m_shader;

    int m_forward;
    int m_sideways;
    int m_vertical;

    int m_lastX;
    int m_lastY;

    bool m_capture;

    bool m_isPaused; // Tracks whether the simulation is paused

    //----- Frame related -----//
    GLuint m_defaultFBO;
    int m_fbo_width;
    int m_fbo_height;
    int m_screen_width;
    int m_screen_height;

    GLuint m_frame_shader;
    GLuint m_fullscreen_vbo;
    GLuint m_fullscreen_vao;
    GLuint m_fbo;
    GLuint m_fbo_texture;
    GLuint m_fbo_renderbuffer;

    void generateScreen();
    void makeFBO();
    void paintFrame(GLuint texture);

    // Device Correction Variables
    int m_devicePixelRatio;


private slots:

    // Physics Tick
    void tick();
};
