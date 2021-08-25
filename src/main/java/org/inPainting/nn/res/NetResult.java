package org.inPainting.nn.res;

import lombok.Getter;
import org.inPainting.utils.ImageLoader;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.inPainting.utils.ImageLoader;


public class NetResult {
    @Getter
    private final INDArray outputPicture;
    @Getter
    private final INDArray score;

    public NetResult(INDArray[] netOutput){
        this.outputPicture = netOutput[1];
        this.score = netOutput[0];
    }

    public double score(){
        return this.score.sumNumber().doubleValue() / this.score.length();
    }

    @Deprecated
    public INDArray mergeByMask(INDArray input, INDArray mask, int width, int height) {
        return ImageLoader.mergeImagesByMask(
                input,
                mask,
                this.outputPicture,
                width,
                height
        );
    }

    @Override
    protected void finalize() {
        outputPicture.close();
        score.close();
    }
}
