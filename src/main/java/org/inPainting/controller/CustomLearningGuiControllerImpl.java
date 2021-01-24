package org.inPainting.controller;

import javafx.fxml.FXML;
import javafx.scene.image.ImageView;
import lombok.Synchronized;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.dataset.MultiDataSet;
import org.inPainting.nn.data.ImageDataSetIterator;
import org.inPainting.nn.data.ImageFileDataSetIterator;
import org.inPainting.nn.GAN;
import org.inPainting.nn.res.NetResult;
import org.inPainting.utils.ImageLoader;
import org.springframework.stereotype.Component;

import java.io.File;
import java.io.IOException;

@Component
@Slf4j
public class CustomLearningGuiControllerImpl implements CustomLearningGuiController {

    @FXML
    private ImageView outputImageView;

    @FXML
    private ImageView realImageView;

    private GAN gan;

    private ImageDataSetIterator trainDataSet;

    private ImageLoader imageLoader = new ImageLoader();

    private NetResult tempOutput;

    @Override
    public void onRefreshGUI() {

        MultiDataSet multiDataSet = trainDataSet.nextRandom();

        int width = 256;
        int height = 256;

        tempOutput = gan.getOutput(multiDataSet.getFeatures()[0]);

        //outputImageView.setImage(imageLoader.drawImage(tempOutput.mergeByMask(multiDataSet.getFeatures()[0],multiDataSet.getFeatures()[1], width, height), width, height));

        outputImageView.setImage(imageLoader.drawImage(tempOutput.getOutputPicture(), width, height));
        realImageView.setImage(imageLoader.drawImage(multiDataSet.getLabels()[0], width, height));

        log.info("Refreshing GUI; Medium Score: " + tempOutput.mediumScore());

        tempOutput = null;
    }

    @Synchronized
    public void reloadData(){
        log.info("Reloading data");
        this.onInitialize();
    }

    @Override
    public long getDataSize(){
        return trainDataSet.getMaxSize();
    }

    @Override
    public void onInitialize() {
        //Switching to storing data in File instead of memory
        trainDataSet = imageLoader.prepareInFileData();
        log.info("Done loading train data");
    }

    @Override
    public void onTrainLoop(long loopNo, boolean t) {

        if (loopNo % (trainDataSet.getMaxSize()*2) == 0){
            try {
                ModelSerializer.writeModel(gan.getDiscriminator(), new File("discriminator.zip"),true);
                ModelSerializer.writeModel(gan.getNetwork(), new File("gan.zip"),true);
            } catch (IOException e) {
                e.printStackTrace();
            }

            log.info("Saving model loopNo="+loopNo);
        }


        if (!trainDataSet.hasNext()) {
            log.info("Resetting ImageDataSetIterator");
            trainDataSet.reset();
            System.gc();
        }

        gan.fit(trainDataSet.next(), t);
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