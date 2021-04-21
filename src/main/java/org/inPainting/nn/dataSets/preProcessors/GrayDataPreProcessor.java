package org.inPainting.nn.dataSets.preProcessors;

import javafx.scene.paint.Color;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.MultiDataSetPreProcessor;

public class GrayDataPreProcessor implements MultiDataSetPreProcessor {
    @Override
    public void preProcess(MultiDataSet multiDataSet) {
        this.toGray(multiDataSet.getFeatures()[0]);
        this.toGray(multiDataSet.getLabels()[0]);
    }

    private void toGray(INDArray rgbImage){
        for (int y = 0; y < rgbImage.shape()[2]; y++) {
            for (int x = 0; x < rgbImage.shape()[3]; x++) {

                Color grayscale = new Color(rgbImage.getDouble(0,0,y,x), rgbImage.getDouble(0,1,y,x), rgbImage.getDouble(0,2,y,x), 1).grayscale();

                rgbImage.putScalar(new int[]{0,0,y,x}, grayscale.getRed());
                rgbImage.putScalar(new int[]{0,1,y,x}, grayscale.getGreen());
                rgbImage.putScalar(new int[]{0,2,y,x}, grayscale.getBlue());
            }
        }
    }
}
