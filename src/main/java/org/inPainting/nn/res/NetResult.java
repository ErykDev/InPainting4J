package org.inPainting.nn.res;

import lombok.Getter;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.inPainting.utils.ImageLoader;


public class NetResult {
    @Getter
    private INDArray outputPicture;
    @Getter
    private double fakeScore;
    @Getter
    private double realScore;

    public NetResult(INDArray[] netOutput){
        this.outputPicture = netOutput[1];
        this.realScore = netOutput[0].getDouble(1);
        this.fakeScore = netOutput[0].getDouble(0);
    }

    public INDArray mergeByMask(INDArray input, INDArray mask, int width, int height){
        return ImageLoader.mergeImagesByMask(
                input,
                mask,
                this.outputPicture,
                width,
                height
        );
    }

    @Override
    protected void finalize() throws Throwable {
        outputPicture.close();
        outputPicture = null;
    }
}
