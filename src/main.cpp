#include "QtCore/qsettings.h"
#include "mainwindow.h"
#include <cstdlib>
#include <ctime>

#include <QApplication>
#include <QSurfaceFormat>
#include <QScreen>
#include <QCommandLineParser>
#include <iostream>

#include <simulation.h>
#include <QCoreApplication>
#include <QCommandLineParser>
#include <QImage>
#include <QtCore>

using namespace Eigen;

int main(int argc, char *argv[])
{
    srand(static_cast<unsigned>(time(0)));

    // Create a Qt application
    QApplication a(argc, argv);
    QCoreApplication::setApplicationName("Simulation");
    QCoreApplication::setOrganizationName("CS 2240");
    QCoreApplication::setApplicationVersion(QT_VERSION_STR);

    // Read and parse a ini file
    QCommandLineParser parser;
    parser.addHelpOption();
    parser.addPositionalArgument("config", "Path of the config file.");
    parser.process(a);

    auto positionalArgs = parser.positionalArguments();
    if (positionalArgs.size() != 1) {
        std::cerr << "Not enough arguments. Please provide a path to a config file (.ini) as a command-line argument." << std::endl;
        a.exit(1);
        return 1;
    }

    QSettings settings( positionalArgs[0], QSettings::IniFormat );
    Settings simSettings;
    simSettings.inputMeshPath = settings.value("IO/Mesh").toString();
    simSettings.obstacleMeshPath = settings.value("IO/ObstacleMesh").toString();
    simSettings.integrateMethod = settings.value("Settings/integrate_method").toInt();
    double gY = settings.value("Settings/g").toDouble();
    simSettings._g = Vector3d(0, gY, 0);
    simSettings._kFloor = settings.value("Settings/kFloor").toDouble();
    simSettings._lambda = settings.value("Settings/lambda").toDouble();
    simSettings._mu = settings.value("Settings/mu").toDouble();
    simSettings._phi = settings.value("Settings/phi").toDouble();
    simSettings._psi = settings.value("Settings/psi").toDouble();
    simSettings._density = settings.value("Settings/density").toDouble();

    simSettings.translation = Eigen::Vector3d(
        settings.value("Transform/TranslationX").toDouble(),
        settings.value("Transform/TranslationY").toDouble(),
        settings.value("Transform/TranslationZ").toDouble()
        );
    simSettings.rotationZ = settings.value("Transform/RotationZ").toDouble();

    simSettings.isCustomizeTimeStep = settings.value("Settings/isCustomizeTimeStep").toBool();
    simSettings.integrationTimeStep = settings.value("Settings/integrationTimeStep").toDouble();
    simSettings.isAdaptiveTimeStep = settings.value("Settings/isAdaptiveTimeStep").toBool();

    simSettings.isParallelize = settings.value("Settings/isParallelize").toBool();

    simSettings.isFBO = settings.value("Settings/isFBO").toBool();
    simSettings.isFXAA = settings.value("Settings/isFXAA").toBool();

    // Set OpenGL version to 4.1 and context to Core
    QSurfaceFormat fmt;
    fmt.setVersion(4, 1);
    fmt.setProfile(QSurfaceFormat::CoreProfile);
    QSurfaceFormat::setDefaultFormat(fmt);

    // Create a GUI window
    MainWindow w(simSettings);
    w.resize(1200, 1000);
    int desktopArea = QGuiApplication::primaryScreen()->size().width() *
                      QGuiApplication::primaryScreen()->size().height();
    int widgetArea = w.width() * w.height();
    if (((float)widgetArea / (float)desktopArea) < 0.75f)
        w.show();
    else
        w.showMaximized();


    return a.exec();
}
