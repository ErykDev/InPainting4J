package org.inPainting.controller;

import org.inPainting.nn.GAN;

public interface CustomLearningGuiController {
    void onRefreshGUI();

    void onInitialize();

    void onTrainLoop(long iteration, int discriminatorFitPause, boolean discriminatorFit);

    void onTestAction();

    GAN onGetGANNetwork();

    void onSetNeuralNetwork(GAN restoreMultiLayerNetwork);

    long getDataSize();
}