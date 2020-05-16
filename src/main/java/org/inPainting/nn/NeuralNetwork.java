package org.inPainting.nn;

import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.graph.ReshapeVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;

import static org.inPainting.utils.LayerUtils.*;

public class NeuralNetwork {

    public static ComputationGraph loadNetworkGraph(File file) throws IOException {
        return ComputationGraph.load(file, true);
    }

    public static ComputationGraph getGenerator() {
        int inputChannels = 5;
        int outputChannels = 4;
        int[] doubleKernel = {2,2};
        int[] doubleStride = {2,2};
        int[] noStride = {1,1};

        return new ComputationGraph(new NeuralNetConfiguration.Builder()
                .updater(Adam.builder().learningRate(GAN.LEARNING_RATE).beta1(GAN.LEARNING_BETA1).build())
                .l2(5*1e-4)
                .graphBuilder()
                .allowDisconnected(true)
                .addInputs("Input")
                //m + rgb 256x256x5x1
                .setInputTypes(InputType.convolutional(256,256,5))

                //Generator
                //Encoder 256x256x5 -> 128x128x20
                .addLayer("GENCNN1",
                        convInitSame(
                                (inputChannels),
                                (inputChannels*4),
                                doubleStride,
                                doubleKernel,
                                Activation.LEAKYRELU),
                        "Input")
                //Encoder 128x128x20 -> 64x64x80
                .addLayer("GENCNN2",
                        convInitSame(
                                (inputChannels*4),
                                (inputChannels*16),
                                doubleStride,
                                doubleKernel,
                                Activation.LEAKYRELU),
                        "GENCNN1")
                //Encoder 64x64x80 -> 32x32x320
                .addLayer("GENCNN3",
                        convInitSame(
                                (inputChannels*16),
                                (inputChannels*64),
                                doubleStride,
                                doubleKernel,
                                Activation.LEAKYRELU),
                        "GENCNN2")
                //Encoder 32x32x320 -> 16x16x1280
                .addLayer("GENCNN4",
                        convInitSame(
                                (inputChannels*64),
                                (inputChannels*256),
                                doubleStride,
                                doubleKernel,
                                Activation.LEAKYRELU),
                        "GENCNN3")

                //Decoder Vertex 16x16x1280 -> 32x32x320x1
                .addVertex("GENRV1", new ReshapeVertex(1,320,32,32),"GENCNN4")
                //Merging Decoder with GENCNN1
                .addVertex("GENmerge1",
                        new MergeVertex(),
                        "GENCNN3","GENRV1")
                //Decoder 32x32x256
                .addLayer("GENCNN5",
                        convInitSame(
                                (inputChannels*64*2),
                                (inputChannels*64),
                                noStride,
                                doubleKernel,
                                Activation.LEAKYRELU),
                        "GENmerge1")
                //Decoder Vertex 32x32x320x1 -> 64x64x80x1
                .addVertex("GENRV2", new ReshapeVertex(1,80,64,64),"GENCNN5")
                //Merging Decoder with Input
                .addVertex("GENmerge2",
                        new MergeVertex(),
                        "GENCNN2","GENRV2")
                //Decoder 64x64x80
                .addLayer("GENCNN6",
                        convInitSame(
                                (inputChannels*16*2),
                                (inputChannels*16),
                                noStride,
                                doubleKernel,
                                Activation.LEAKYRELU),
                        "GENmerge2")
                //Decoder Vertex 64x64x80 -> 128x128x20
                .addVertex("GENRV3", new ReshapeVertex(1,20,128,128),"GENCNN6")
                //Merging Decoder with Input
                .addVertex("GENmerge3",
                        new MergeVertex(),
                        "GENCNN1","GENRV3")
                //Decoder 128x128x20
                .addLayer("GENCNN7",
                        convInitSame(
                                (inputChannels*4*2),
                                (inputChannels*4),
                                noStride,
                                doubleKernel,
                                Activation.LEAKYRELU),
                        "GENmerge3")
                //Decoder Vertex 128x128x20 -> 256x256x5
                .addVertex("GENRV4", new ReshapeVertex(1,5,256,256),"GENCNN7")
                //Merging Decoder with Input
                .addVertex("GENmerge4",
                        new MergeVertex(),
                        "Input","GENRV4")
                //Decoder 256x256x4
                .addLayer("GENCNN8",
                        convInitSame(
                                (inputChannels*2),
                                (outputChannels),
                                Activation.LEAKYRELU),
                        "GENmerge4")
                //Decoder Loss
                .addLayer("GENCNNloss", new CnnLossLayer.Builder(LossFunctions.LossFunction.XENT)
                        .activation(Activation.SIGMOID)
                        .build(),"GENCNN8")

                .setOutputs("GENCNNloss")
                .build()
        );
    }

    public static ComputationGraph getDiscriminator() {
        int seed = 123;
        int channels = 4;
        double nonZeroBias = 1;
        return new ComputationGraph(new NeuralNetConfiguration.Builder()
                .weightInit(new NormalDistribution(0.0,1E-1))
                .updater(new Nesterovs(1e-2,9E-1))
                .biasUpdater(new Nesterovs(2e-2,9E-1))
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
                .l2(5*1e-4)
                .miniBatch(false)
                .activation(Activation.RELU)
                .seed(seed)
                .graphBuilder()

                .addInputs("Input")
                //rgb 256x256
                .setInputTypes(InputType.convolutional(256,256,3))

                .addLayer("DISCNN1", new ConvolutionLayer.Builder(new int[]{11,11}, new int[]{4, 4})
                        .cudnnAlgoMode(ConvolutionLayer.AlgoMode.PREFER_FASTEST)
                        .convolutionMode(ConvolutionMode.Truncate)
                        .nIn(channels)
                        .nOut(96)
                        .build(),"Input")
                .addLayer("DISLRN1", new LocalResponseNormalization.Builder().build(),"DISCNN1")
                .addLayer("DISSL1", new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(3,3)
                        .stride(2,2)
                        .padding(1,1)
                        .build(),"DISLRN1")
                .addLayer("DISCNN2", new ConvolutionLayer.Builder(new int[]{5,5}, new int[]{1,1}, new int[]{2,2})
                        .cudnnAlgoMode(ConvolutionLayer.AlgoMode.PREFER_FASTEST)
                        .convolutionMode(ConvolutionMode.Truncate)
                        .nOut(256)
                        .biasInit(nonZeroBias)
                        .build(),"DISSL1")
                .addLayer("DISSL2", new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{3, 3}, new int[]{2, 2})
                        .convolutionMode(ConvolutionMode.Truncate)
                        .build(),"DISCNN2")
                .addLayer("DISLRN2", new LocalResponseNormalization.Builder().build(),"DISSL2")
                .addLayer("DISCNN3", new ConvolutionLayer.Builder()
                        .kernelSize(3,3)
                        .stride(1,1)
                        .convolutionMode(ConvolutionMode.Same)
                        .cudnnAlgoMode(ConvolutionLayer.AlgoMode.PREFER_FASTEST)
                        .nOut(384)
                        .build(),"DISLRN2")
                .addLayer("DISCNN4", new ConvolutionLayer.Builder(new int[]{3,3}, new int[]{1,1})
                        .cudnnAlgoMode(ConvolutionLayer.AlgoMode.PREFER_FASTEST)
                        .nOut(384)
                        .biasInit(nonZeroBias)
                        .build(),"DISCNN3")
                .addLayer("DISCNN5", new ConvolutionLayer.Builder(new int[]{3,3}, new int[]{1,1})
                        .cudnnAlgoMode(ConvolutionLayer.AlgoMode.PREFER_FASTEST)
                        .nOut(256)
                        .biasInit(nonZeroBias)
                        .build(),"DISCNN4")
                .addLayer("DISSL3", new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{3,3}, new int[]{2,2})
                        .convolutionMode(ConvolutionMode.Truncate)
                        .build(),"DISCNN5")
                .addLayer("DISFFN1", new DenseLayer.Builder()
                        .weightInit(new NormalDistribution(0, 5E-3))
                        .nOut(256)
                        .biasInit(nonZeroBias)
                        .build(),"DISSL3")
                .addLayer("DISFFN2", new DenseLayer.Builder()
                        .nOut(256)
                        .weightInit(new NormalDistribution(0, 5E-3))
                        .biasInit(nonZeroBias)
                        .dropOut(0.5)
                        .build(),"DISFFN1")
                .addLayer("DISFFOUT", new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(2)
                        .activation(Activation.SOFTMAX)
                        .weightInit(new NormalDistribution(0, 0.005))
                        .biasInit(0.1)
                        .build(),"DISFFN2")

                .setOutputs("DISFFOUT")
                .build()
        );
    }
}