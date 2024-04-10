#pragma once

#include <QMainWindow>
#include "glwidget.h"

class MainWindow : public QWidget
{
    Q_OBJECT

public:
//    MainWindow();
    explicit MainWindow(const Settings& settings, QWidget *parent = nullptr); // Adjusted constructor
    ~MainWindow();

private:

    GLWidget *glWidget;
};
