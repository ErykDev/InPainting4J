package org.inPainting.controller;

import org.inPainting.nn.GAN;

public interface CustomLearningGuiController {
    void onRefreshGUI();

    void onInitialize();

    void onTrainLoop(long loopNo,boolean trainD);

    void onTestAction();

    void onSetNeuralNetwork(GAN restoreMultiLayerNetwork);

    long getDataSize();
}