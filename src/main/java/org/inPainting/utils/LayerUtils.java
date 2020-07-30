package org.inPainting.utils;

import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.graph.ReshapeVertex;
import org.deeplearning4j.nn.conf.layers.Convolution2D;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.Deconvolution2D;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.nd4j.linalg.activations.Activation;

public class LayerUtils {
    private LayerUtils() {
        // This is intentionally empty
    }

    @SuppressWarnings("SameParameterValue")
    public static ConvolutionLayer convInit(String name, int in, int out, int[] kernel, int[] stride, int[] pad, double bias) {
        return new ConvolutionLayer.Builder(kernel, stride, pad).name(name).nIn(in).nOut(out).biasInit(bias).build();
    }

    public static ConvolutionLayer conv3x3(String name, int out, double bias) {
        return new ConvolutionLayer.Builder(new int[]{3,3}, new int[] {1,1}, new int[] {1,1}).name(name).nOut(out).biasInit(bias).build();
    }

    public static ReshapeVertex reshape(int... newShape){
        return new ReshapeVertex(newShape);
    }

    @SuppressWarnings("SameParameterValue")
    public static ConvolutionLayer convInitSame(int in, int out, Activation activation) {
        return new Convolution2D.Builder(1,1).convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(ConvolutionLayer.AlgoMode.PREFER_FASTEST).activation(activation).stride(1,1).nIn(in).nOut(out).build();
    }

    public static ConvolutionLayer convInit(int in, int out,int[] stride, int[] kernel,Activation activation) {
        return new Convolution2D.Builder(kernel).cudnnAlgoMode(ConvolutionLayer.AlgoMode.PREFER_FASTEST).activation(activation).stride(stride).nIn(in).nOut(out).build();
    }

    @SuppressWarnings("SameParameterValue")
    public static Deconvolution2D deConvolution2D(int in, int out, int[] stride, int[] kernel, Activation activation) {
        return new Deconvolution2D.Builder(kernel).convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(ConvolutionLayer.AlgoMode.PREFER_FASTEST).activation(activation).stride(stride).nIn(in).nOut(out).build();
    }


    @SuppressWarnings("SameParameterValue")
    public static ConvolutionLayer convInitSame( int in, int out,int[] stride, int[] kernel,Activation activation) {
        return new Convolution2D.Builder(kernel).convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(ConvolutionLayer.AlgoMode.PREFER_FASTEST).activation(activation).stride(stride).nIn(in).nOut(out).build();
    }

    public static SubsamplingLayer maxPool(String name, int[] kernel) {
        return new SubsamplingLayer.Builder(kernel, new int[]{2,2}).name(name).build();
    }
}
