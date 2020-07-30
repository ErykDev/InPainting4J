package org.inPainting.controller;

import javafx.fxml.FXML;
import javafx.scene.image.ImageView;
import javafx.scene.paint.Color;
import org.deeplearning4j.util.ModelSerializer;
import org.inPainting.nn.GAN;
import org.inPainting.nn.ImageDataSetIterator;
import org.inPainting.nn.res.NetResult;
import org.inPainting.utils.ImageUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.MultiDataSet;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;

import java.io.File;
import java.io.IOException;
import java.util.Random;

@Component
public class CustomLearningGuiControllerImpl implements CustomLearningGuiController {
    protected final Logger log = LoggerFactory.getLogger(this.getClass());

    @FXML
    private ImageView outputImageView;

    @FXML
    private ImageView realImageView;

    private GAN gan;

    private ImageDataSetIterator trainDataSet;

    private Random r = new Random();

    @Override
    public void onRefreshGUI() {

        MultiDataSet multiDataSet = trainDataSet.nextRandom();

        int width = 256;
        int height = 256;

        INDArray input = multiDataSet.getFeatures()[0];
        INDArray real = multiDataSet.getLabels()[0];

        NetResult netResult = gan.getOutput(input);

        //INDArray mergedOutput = netResult.mergeByMask(input, width, height);

        outputImageView.setImage(ImageUtils.emptyImage(Color.BLACK, width, height));
        outputImageView.setImage(ImageUtils.drawImage(netResult.getOutput(), width, height));
        realImageView.setImage(ImageUtils.drawImage(real, width, height));

        log.info("Refreshing GUI; Result Score: " + netResult.getRealScore()+";");
    }

    @Override
    public void onInitialize() {
        // loading training data
        trainDataSet = ImageUtils.prepareData();
        log.info("Done loading train data");
        System.gc();
    }


    @Override
    public void onTrainLoop(long loopNo, boolean t) {

        if (!trainDataSet.hasNext()) {
            log.info("Resetting ImageDataSetIterator");
            trainDataSet.reset();
            System.gc();
        }

        if (loopNo % 14 == 0 && loopNo != 0)
            gan.fit(trainDataSet.next(), t);
        else
            gan.fit(trainDataSet.next(), false);
    }

    @Override
    public void onTestAction() {
        this.onRefreshGUI();
    }

    @Override
    public GAN onGetGANNetwork() {
        return this.gan;
    }

    @Override
    public void onSetNeuralNetwork(GAN restoreMultiLayerNetwork) {
        this.gan = restoreMultiLayerNetwork;
    }

}