package org.inPainting.controller;

import javafx.fxml.FXML;
import javafx.scene.image.ImageView;
import javafx.scene.paint.Color;
import org.inPainting.nn.GAN;
import org.inPainting.utils.ImageUtils;
import org.nd4j.linalg.dataset.MultiDataSet;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;

import java.util.Random;

@Component
public class CustomLearningGuiControllerImpl implements CustomLearningGuiController {
    protected final Logger log = LoggerFactory.getLogger(this.getClass());

    @FXML
    private ImageView outputImageView;

    @FXML
    private ImageView realImageView;

    private GAN gan;

    private MultiDataSet[] trainDataSet;

    private Random r = new Random();

    @Override
    public void onRefreshGUI() {
    }

    @Override
    public void onInitialize() {
        // loading training data
        trainDataSet = ImageUtils.prepareData();
        log.info("Done loading train data");
        System.gc();
    }


    @Override
    public void onTrainLoop(long loopNo,boolean t) {
        gan.fit(trainDataSet[r.nextInt(trainDataSet.length)],t);
    }

    @Override
    public void onTestAction() {
        int dataPos = r.nextInt(trainDataSet.length);

        outputImageView.setImage(ImageUtils.emptyImage(Color.BLACK,256,256));
        outputImageView.setImage(gan.getOutput(trainDataSet[dataPos].getFeatures()[0],256,256));
        realImageView.setImage(ImageUtils.drawImage(trainDataSet[dataPos].getLabels()[0],256,256));
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