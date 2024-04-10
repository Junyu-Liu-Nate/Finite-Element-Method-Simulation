#include "mainwindow.h"
#include <QHBoxLayout>

//MainWindow::MainWindow()
//{
//    glWidget = new GLWidget();

//    QHBoxLayout *container = new QHBoxLayout;
//    container->addWidget(glWidget);
//    this->setLayout(container);
//}

MainWindow::MainWindow(const Settings& settings, QWidget *parent) {
    glWidget = new GLWidget(settings);

        QHBoxLayout *container = new QHBoxLayout;
        container->addWidget(glWidget);
        this->setLayout(container);
}

MainWindow::~MainWindow()
{
    delete glWidget;
}
