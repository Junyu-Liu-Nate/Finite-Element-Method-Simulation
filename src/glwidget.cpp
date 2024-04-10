#include "glwidget.h"

#include <QApplication>
#include <QKeyEvent>
#include <iostream>

#include "graphics/shaderLoader.h"

#define SPEED 1.5
#define ROTATE_SPEED 0.0025

using namespace std;

GLWidget::GLWidget(const Settings& settings, QWidget *parent) :
    QOpenGLWidget(parent),
    m_deltaTimeProvider(),
    m_intervalTimer(),
    m_sim(settings),
    m_camera(),
    m_shader(),
    m_forward(),
    m_sideways(),
    m_vertical(),
    m_lastX(),
    m_lastY(),
    m_capture(false),
    m_isPaused(true) // Start with the simulation paused
{
    m_fbo_width = this->width();
    m_fbo_height = this->height();

    // GLWidget needs all mouse move events, not just mouse drag events
    setMouseTracking(true);

    // Hide the cursor since this is a fullscreen app
    QApplication::setOverrideCursor(Qt::ArrowCursor);

    // GLWidget needs keyboard focus
    setFocusPolicy(Qt::StrongFocus);

    // Function tick() will be called once per interva
    connect(&m_intervalTimer, SIGNAL(timeout()), this, SLOT(tick()));
}

GLWidget::~GLWidget()
{
    if (m_shader != nullptr) delete m_shader;
}

// ================== Basic OpenGL Overrides

void GLWidget::initializeGL()
{
    m_devicePixelRatio = this->devicePixelRatio();

    m_defaultFBO = 2;
    m_screen_width = size().width() * devicePixelRatio();
    m_screen_height = size().height() * devicePixelRatio();
    m_fbo_width = m_screen_width;
    m_fbo_height = m_screen_height;

    // Initialize GL extension wrangler
    glewExperimental = GL_TRUE;
    GLenum err = glewInit();
    if (err != GLEW_OK) fprintf(stderr, "Error while initializing GLEW: %s\n", glewGetErrorString(err));
    fprintf(stdout, "Successfully initialized GLEW %s\n", glewGetString(GLEW_VERSION));

    // Set clear color to white
    glClearColor(1, 1, 1, 1);

    // Enable depth-testing and backface culling
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);

    glViewport(0, 0, size().width() * m_devicePixelRatio, size().height() * m_devicePixelRatio);

    // Initialize the shader and simulation
    m_shader = new Shader(":/resources/shaders/shader.vert", ":/resources/shaders/shader.frag");
    m_sim.init();

    // Initialize camera with a reasonable transform
    Eigen::Vector3f eye    = {0, 2, -5};
    Eigen::Vector3f target = {0, 1,  0};
    m_camera.lookAt(eye, target);
    m_camera.setOrbitPoint(target);
    m_camera.setPerspective(120, width() / static_cast<float>(height()), 0.1, 50);

    m_deltaTimeProvider.start();
    m_intervalTimer.start(1000 / 60);

    // ====== Texture shader - operates on FBO
    m_frame_shader = ShaderLoader::createShaderProgram("./resources/shaders/frame.vert", "./resources/shaders/frame.frag");
    if (m_frame_shader == 0) {
        std::cerr << "Failed to create frame shader program." << std::endl;
    }

    generateScreen();
    makeFBO();
}

void GLWidget::generateScreen() {
    std::vector<GLfloat> fullscreen_quad_data = {
        // Positions    // UV Coordinates
        -1.0f,  1.0f, 0.0f,   0.0f, 1.0f, // Top Left
        -1.0f, -1.0f, 0.0f,   0.0f, 0.0f, // Bottom Left
        1.0f, -1.0f, 0.0f,   1.0f, 0.0f,  // Bottom Right
        1.0f,  1.0f, 0.0f,   1.0f, 1.0f,  // Top Right
        -1.0f,  1.0f, 0.0f,   0.0f, 1.0f, // Top Left
        1.0f, -1.0f, 0.0f,   1.0f, 0.0f   // Bottom Right
    };

    // Generate and bind a VBO and a VAO for a fullscreen quad
    glGenBuffers(1, &m_fullscreen_vbo);
    glBindBuffer(GL_ARRAY_BUFFER, m_fullscreen_vbo);
    glBufferData(GL_ARRAY_BUFFER, fullscreen_quad_data.size()*sizeof(GLfloat), fullscreen_quad_data.data(), GL_STATIC_DRAW);
    glGenVertexArrays(1, &m_fullscreen_vao);
    glBindVertexArray(m_fullscreen_vao);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(GLfloat), nullptr);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(GLfloat), (const void*)(3 * sizeof(GLfloat)));

    // Unbind the fullscreen quad's VBO and VAO
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}

void GLWidget::makeFBO(){
    // Generate and bind an empty texture, set its min/mag filter interpolation, then unbind
    glGenTextures(1, &m_fbo_texture); // Generate fbo texture
    glActiveTexture(GL_TEXTURE0); // Set the active texture slot to texture slot 0
    glBindTexture(GL_TEXTURE_2D, m_fbo_texture); // Bind fbo texture

    // Load empty image into fbo texture
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, m_fbo_width, m_fbo_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

    // Set min and mag filters' interpolation mode to linear
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glBindTexture(GL_TEXTURE_2D, 0); // Unbind fbo texture

    // Generate and bind a renderbuffer of the right size, set its format, then unbind
    glGenRenderbuffers(1, &m_fbo_renderbuffer);
    glBindRenderbuffer(GL_RENDERBUFFER, m_fbo_renderbuffer);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, m_fbo_width, m_fbo_height);
    glBindRenderbuffer(GL_RENDERBUFFER, 0);

    // Generate and bind an FBO
    glGenFramebuffers(1, &m_fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, m_fbo);

    // Add our texture as a color attachment, and our renderbuffer as a depth+stencil attachment, to our FBO
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m_fbo_texture, 0);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, m_fbo_renderbuffer);

    // Unbind the FBO
    glBindFramebuffer(GL_FRAMEBUFFER, m_defaultFBO);
}

void GLWidget::paintGL()
{
    // Bind FBO
    glBindFramebuffer(GL_FRAMEBUFFER, m_fbo);

    // Call glViewport
    glViewport(0, 0, m_screen_width, m_screen_height);

    // Clear screen color and depth before painting
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    //----- Original code
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    m_shader->bind();
    m_shader->setUniform("proj", m_camera.getProjection());
    m_shader->setUniform("view", m_camera.getView());
    m_sim.draw(m_shader);
    m_shader->unbind();
    //--------------------//

    // Bind the default framebuffer
    glBindFramebuffer(GL_FRAMEBUFFER, m_defaultFBO);

    glViewport(0, 0, m_screen_width, m_screen_height);

    // Clear the color and depth buffers
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    paintFrame(m_fbo_texture);
}

void GLWidget::paintFrame(GLuint texture)
{
    glUseProgram(m_frame_shader);

    glUniform1i(glGetUniformLocation(m_frame_shader, "isFBO"), m_sim.m_settings.isFBO);
    glUniform1i(glGetUniformLocation(m_frame_shader, "isFXAA"), m_sim.m_settings.isFXAA);

    glUniform1f(glGetUniformLocation(m_frame_shader, "screenWidth"), m_screen_width);
    glUniform1f(glGetUniformLocation(m_frame_shader, "screenHeight"), m_screen_height);

    glBindVertexArray(m_fullscreen_vao);

    // Bind "texture" to slot 0
    glActiveTexture(GL_TEXTURE0);           // Activate texture slot 0
    glBindTexture(GL_TEXTURE_2D, texture);  // Bind the texture to the active texture slot

    // Note that the textureImg uniform is implicitly set to use texture slot 0
    // because GL_TEXTURE0 is active when the texture is bound.

    glDrawArrays(GL_TRIANGLES, 0, 6);
    glBindTexture(GL_TEXTURE_2D, 0);
    glBindVertexArray(0);
    glUseProgram(0);
}

void GLWidget::resizeGL(int w, int h)
{
    glViewport(0, 0, w, h);
    m_camera.setAspect(static_cast<float>(w) / h);
}

// ================== Event Listeners

void GLWidget::mousePressEvent(QMouseEvent *event)
{
    m_capture = true;
    m_lastX = event->position().x();
    m_lastY = event->position().y();
}

void GLWidget::mouseMoveEvent(QMouseEvent *event)
{
    if (!m_capture) return;

    int currX  = event->position().x();
    int currY  = event->position().y();

    int deltaX = currX - m_lastX;
    int deltaY = currY - m_lastY;

    if (deltaX == 0 && deltaY == 0) return;

    m_camera.rotate(deltaY * ROTATE_SPEED,
                    -deltaX * ROTATE_SPEED);

    m_lastX = currX;
    m_lastY = currY;
}

void GLWidget::mouseReleaseEvent(QMouseEvent *event)
{
    m_capture = false;
}

void GLWidget::wheelEvent(QWheelEvent *event)
{
    float zoom = 1 - event->pixelDelta().y() * 0.1f / 120.f;
    m_camera.zoom(zoom);
}

void GLWidget::keyPressEvent(QKeyEvent *event)
{
    if (event->isAutoRepeat()) return;

    switch (event->key())
    {
    case Qt::Key_Space: // Toggle pause/resume simulation
        m_isPaused = !m_isPaused;
        break;
    case Qt::Key_W: m_forward  += SPEED; break;
    case Qt::Key_S: m_forward  -= SPEED; break;
    case Qt::Key_A: m_sideways -= SPEED; break;
    case Qt::Key_D: m_sideways += SPEED; break;
    case Qt::Key_F: m_vertical -= SPEED; break;
    case Qt::Key_R: m_vertical += SPEED; break;
    case Qt::Key_C: m_camera.toggleIsOrbiting(); break;
    case Qt::Key_T: m_sim.toggleWire(); break;
    case Qt::Key_Escape: QApplication::quit();
    }
}

void GLWidget::keyReleaseEvent(QKeyEvent *event)
{
    if (event->isAutoRepeat()) return;

    switch (event->key())
    {
    case Qt::Key_W: m_forward  -= SPEED; break;
    case Qt::Key_S: m_forward  += SPEED; break;
    case Qt::Key_A: m_sideways += SPEED; break;
    case Qt::Key_D: m_sideways -= SPEED; break;
    case Qt::Key_F: m_vertical += SPEED; break;
    case Qt::Key_R: m_vertical -= SPEED; break;
    }
}

// ================== Physics Tick

void GLWidget::tick()
{
    float deltaSeconds = m_deltaTimeProvider.restart() / 1000.f;

    if (!m_isPaused) {
        m_sim.update(deltaSeconds);
    }

    // Move camera
    auto look = m_camera.getLook();
    look.y() = 0;
    look.normalize();
    Eigen::Vector3f perp(-look.z(), 0, look.x());
    Eigen::Vector3f moveVec = m_forward * look.normalized() + m_sideways * perp.normalized() + m_vertical * Eigen::Vector3f::UnitY();
    moveVec *= deltaSeconds;
    m_camera.move(moveVec);

    // Flag this view for repainting (Qt will call paintGL() soon after)
    update();
}
