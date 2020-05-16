package org.inPainting.utils;

import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.layers.Convolution2D;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.nd4j.linalg.activations.Activation;

public class LayerUtils {
    private LayerUtils() {
        // This is intentionally empty
    }

    @SuppressWarnings("SameParameterValue")
    public static ConvolutionLayer convInitSame(int in, int out, Activation activation) {
        return new Convolution2D.Builder(1,1).convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(ConvolutionLayer.AlgoMode.PREFER_FASTEST).activation(activation).stride(1,1).nIn(in).nOut(out).build();
    }

    @SuppressWarnings("SameParameterValue")
    public static ConvolutionLayer convInitSame(int in, int out,int[] stride, int[] kernel, Activation activation) {
        return new Convolution2D.Builder(kernel).convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(ConvolutionLayer.AlgoMode.PREFER_FASTEST).activation(activation).stride(stride).nIn(in).nOut(out).build();
    }
}
