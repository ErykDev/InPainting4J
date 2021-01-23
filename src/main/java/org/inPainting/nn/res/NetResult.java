package org.inPainting.nn.res;

import lombok.Getter;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.inPainting.utils.ImageLoader;


public class NetResult {
    @Getter
    private INDArray outputPicture;
    @Getter
    private INDArray score;

    public NetResult(INDArray[] netOutput){
        this.outputPicture = netOutput[1];
        this.score = netOutput[0];
    }

    public double mediumScore(){
        return this.score.medianNumber().doubleValue();
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
        score.close();
        outputPicture = null;
        score = null;
    }
}
