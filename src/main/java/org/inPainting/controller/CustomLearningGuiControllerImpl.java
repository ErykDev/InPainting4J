package org.inPainting.controller;

import javafx.fxml.FXML;
import javafx.scene.image.ImageView;
import lombok.Synchronized;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.dataset.MultiDataSet;
import org.inPainting.nn.dataSets.ImageDataSetIterator;
import org.inPainting.nn.GAN;
import org.inPainting.nn.res.NetResult;
import org.inPainting.utils.ImageLoader;
import org.springframework.stereotype.Component;

@Component
@Slf4j
public class CustomLearningGuiControllerImpl implements CustomLearningGuiController {

    @FXML
    private ImageView outputImageView;

    @FXML
    private ImageView realImageView;

    private GAN gan;

    private ImageDataSetIterator trainDataSet;

    private final ImageLoader imageLoader = new ImageLoader();

    private NetResult tempOutput;

    @Override
    public void onRefreshGUI() {

        MultiDataSet multiDataSet = trainDataSet.nextRandom();

        int width = 256;
        int height = 256;

        tempOutput = gan.getOutput(multiDataSet.getFeatures());

        outputImageView.setImage(imageLoader.drawImage(tempOutput.getOutputPicture(), width, height));
        realImageView.setImage(imageLoader.drawImage(multiDataSet.getLabels()[0], width, height));

        log.info("Refreshing GUI; Positive Score: " + tempOutput.score());

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
        log.info("Done loading data");
    }

    @Override
    public void onTrainLoop(long loopNo, boolean t) {

        if (!trainDataSet.hasNext()) {
            log.info("Resetting ImageDataSetIterator");
            trainDataSet.reset();
            System.gc();
        }

        if (loopNo % 4 == 0)
            gan.fit(trainDataSet.next(), t);
        else
            gan.fit(trainDataSet.next(), false);
    }

    @Override
    public void onTestAction() {
        this.onRefreshGUI();
    }

    @Override
    public void onSetNeuralNetwork(GAN restoreMultiLayerNetwork) {
        this.gan = restoreMultiLayerNetwork;
    }
}