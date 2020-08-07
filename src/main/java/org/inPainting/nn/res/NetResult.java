package org.inPainting.nn.res;

import javafx.scene.image.WritableImage;
import lombok.Getter;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.inPainting.utils.ImageUtils;

public class NetResult {
    @Getter
    private INDArray _outputPicture;
    @Getter
    private double _fakeScore;
    @Getter
    private double _realScore;

    public NetResult(INDArray[] netOutput){
        this._outputPicture = netOutput[1];
        this._realScore = netOutput[0].getDouble(1);
        this._fakeScore = netOutput[0].getDouble(0);
    }

    public INDArray mergeByMask(INDArray inputWithMask, int width, int height){
        return ImageUtils.mergeImagesByMask(
                inputWithMask,
                this._outputPicture,
                width,
                height
        );
    }

    public WritableImage drawPicture(int width, int height) {
        return ImageUtils.drawImage(this._outputPicture, width, height);
    }
}
