#-------------------------------------------------
#
# Project created by QtCreator 2017-02-12T04:33:56
#
#-------------------------------------------------

QT       += core

QT       -= gui

TARGET = trainHOG2
CONFIG   += console
CONFIG   -= app_bundle

TEMPLATE = app


SOURCES += main.cpp \
    svmlight/svm_common.c \
    svmlight/svm_learn.c \
    pr_loqo/pr_loqo.c \
    svmlight/svm_hideo.c

INCLUDEPATH += .
INCLUDEPATH += /usr/local/include
INCLUDEPATH += /usr/local/include/opencv
INCLUDEPATH += /usr/local/include/opencv2
INCLUDEPATH += /usr/local/include/opencv2/imgproc
INCLUDEPATH += /usr/local/include/opencv2/highgui
INCLUDEPATH += /usr/local/include/opencv2/ml

LIBS += `pkg-config opencv --cflags --libs`

HEADERS += \
    libsvm/libsvm.h \
    svmlight/kernel.h \
    svmlight/svm_common.h \
    svmlight/svm_learn.h \
    svmlight/svmlight.h \
    pr_loqo/pr_loqo.h

OTHER_FILES += \
    Makefile
