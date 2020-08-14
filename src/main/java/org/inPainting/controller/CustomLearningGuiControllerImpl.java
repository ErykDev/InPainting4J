package org.inPainting.controller;

import javafx.fxml.FXML;
import javafx.scene.image.ImageView;
import lombok.Synchronized;
import lombok.extern.slf4j.Slf4j;
import org.inPainting.nn.GAN;
import org.inPainting.nn.data.ImageMemoryDataSetIterator;
import org.inPainting.nn.res.NetResult;
import org.inPainting.utils.ImageLoader;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.MultiDataSet;
import org.springframework.stereotype.Component;

@Component
@Slf4j
public class CustomLearningGuiControllerImpl implements CustomLearningGuiController {

    @FXML
    private ImageView outputImageView;

    @FXML
    private ImageView realImageView;

    private GAN gan;

    private ImageMemoryDataSetIterator trainDataSet;

    private ImageLoader imageLoader = new ImageLoader(GAN._NetInputShape);

    @Override
    @Synchronized
    public void onRefreshGUI() {

        MultiDataSet multiDataSet = trainDataSet.nextRandom();

        int width = 256;
        int height = 256;

        INDArray input = multiDataSet.getFeatures()[0];
        INDArray real = multiDataSet.getLabels()[0];

        NetResult netResult = gan.getOutput(input);

        //INDArray mergedOutput = netResult.mergeByMask(input, width, height);

        outputImageView.setImage(imageLoader.drawImage(netResult.getOutputPicture(), width, height));
        realImageView.setImage(imageLoader.drawImage(real, width, height));

        log.info("Refreshing GUI; Result Score: " + netResult.getRealScore()+";");

        netResult = null;
    }

    @Override
    public void onInitialize() {
        // loading training data
        trainDataSet = imageLoader.prepareInMemoryData();
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

    @Override
    public long getDataSize(){
        return trainDataSet.getMaxSize();
    }

}