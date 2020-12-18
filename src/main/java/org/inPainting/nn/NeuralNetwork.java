package org.inPainting.nn;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.inPainting.nn.entry.*;

import java.io.File;
import java.io.IOException;

import static org.inPainting.utils.LayerUtils.convInitSame;
import static org.inPainting.utils.LayerUtils.reshape;

public class NeuralNetwork {

    public static void saveNetworkGraph(ComputationGraph neuralNetwork, File file) {
        try {
            ModelSerializer.writeModel(neuralNetwork, file, true);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static ComputationGraph loadNetworkGraph(File file) throws IOException {
        return ComputationGraph.load(file, true);
    }

    public static LEntry[] genLayers() {

        int[] _MergedNetInputShape = {1,4,256,256};

        double nonZeroBias = 1;
        int inputChannels = 4;
        int outputChannels = 3;
        int[] doubleKernel = {2,2};
        int[] doubleStride = {2,2};
        int[] noStride = {1,1};


        return new LEntry[]{
                new VertexEntry("InputGENmerge0", new MergeVertex(), "Input","Mask"),
                new LayerEntry("GENCNN1",
                        convInitSame(
                                ((inputChannels)),
                                ((inputChannels*4)),
                                doubleStride,
                                doubleKernel,
                                Activation.LEAKYRELU),
                        "InputGENmerge0"),

                new LayerEntry("GENCNN2",
                        convInitSame(
                                (inputChannels*4),
                                (inputChannels*16),
                                doubleStride,
                                doubleKernel,
                                Activation.LEAKYRELU),
                        "GENCNN1"),

                new LayerEntry("GENCNN3",
                        convInitSame(
                                (inputChannels*16),
                                (inputChannels*64),
                                doubleStride,
                                doubleKernel,
                                Activation.LEAKYRELU),
                        "GENCNN2"),

                new LayerEntry("GENCNN4",
                        convInitSame(
                                (inputChannels*64),
                                (inputChannels*256),
                                doubleStride,
                                doubleKernel,
                                Activation.LEAKYRELU),
                        "GENCNN3"),

                //Encoder 16x16x1024 -> 8x8x4096
                new LayerEntry("GENCNN5",
                        convInitSame(
                                (inputChannels*256),
                                (inputChannels*1024),
                                doubleStride,
                                doubleKernel,
                                Activation.LEAKYRELU),
                        "GENCNN4"),
                //Decoder Vertex 8x8x4096 -> 16x16x1024
                new VertexEntry("GENRV1",
                        reshape(_MergedNetInputShape[0],_MergedNetInputShape[1]*256,_MergedNetInputShape[2]/16,_MergedNetInputShape[3]/16),
                        "GENCNN5"),
                //Merging Decoder with GENCNN1
                new VertexEntry("GENmerge1",
                        new MergeVertex(),
                        "GENCNN4","GENRV1"),
                //Decoder 16x16x1024
                new LayerEntry("GENCNN6",
                        convInitSame(
                                (inputChannels*256*2),
                                (inputChannels*256),
                                noStride,
                                doubleKernel,
                                Activation.LEAKYRELU),
                        "GENmerge1"),
                //Decoder Vertex 16x16x1024 -> 32x32x256x1
                new VertexEntry("GENRV2",
                        reshape(_MergedNetInputShape[0],_MergedNetInputShape[1]*64,_MergedNetInputShape[2]/8,_MergedNetInputShape[3]/8),
                        "GENCNN6"),

                //Merging Decoder with Input
                new VertexEntry("GENmerge2",
                        new MergeVertex(),
                        "GENCNN3","GENRV2"),
                //Decoder 32x32x256
                new LayerEntry("GENCNN7",
                        convInitSame(
                                (inputChannels*64*2),
                                (inputChannels*64),
                                noStride,
                                doubleKernel,
                                Activation.LEAKYRELU),
                        "GENmerge2"),
                //Decoder Vertex 32x32x256 -> 64x64x64
                new VertexEntry("GENRV3",
                        reshape(_MergedNetInputShape[0],_MergedNetInputShape[1]*16,_MergedNetInputShape[2]/4,_MergedNetInputShape[3]/4),
                        "GENCNN7"),
                //Merging Decoder with Input
                new VertexEntry("GENmerge3",
                        new MergeVertex(),
                        "GENCNN2","GENRV3"),

                //Decoder 64x64x64
                new LayerEntry("GENCNN8",
                        convInitSame(
                                (inputChannels*16*2),
                                (inputChannels*16),
                                noStride,
                                doubleKernel,
                                Activation.LEAKYRELU),
                        "GENmerge3"),
                //Decoder Vertex 64x64x64x1 -> 128x128x*16
                new VertexEntry("GENRV4",
                        reshape(_MergedNetInputShape[0],_MergedNetInputShape[1]*4,_MergedNetInputShape[2]/2,_MergedNetInputShape[3]/2),
                        "GENCNN8"),

                //Merging Decoder with Input
                new VertexEntry("GENmerge4",
                        new MergeVertex(),
                        "GENCNN1","GENRV4"),

                //Decoder 128x128x16x1
                new LayerEntry("GENCNN9",
                        convInitSame(
                                (inputChannels*4*2),
                                (inputChannels*4),
                                Activation.LEAKYRELU),
                        "GENmerge4"),
                //Decoder Vertex 128x128x*16 -> 256x256x4
                new VertexEntry("GENRV5",
                        reshape(_MergedNetInputShape[0],_MergedNetInputShape[1],_MergedNetInputShape[2],_MergedNetInputShape[3]),
                        "GENCNN9"),

                //Merging Decoder with Input
                new VertexEntry("GENmerge5",
                        new MergeVertex(),
                        "InputGENmerge0","GENRV5"),

                //Decoder 256x256x3
                new LayerEntry("GENCNN10",
                        convInitSame(
                                (inputChannels*2),
                                (outputChannels),
                                Activation.LEAKYRELU),
                        "GENmerge5"),

                new LayerEntry("GENCNNLoss", new CnnLossLayer.Builder(LossFunctions.LossFunction.XENT)
                        .activation(Activation.SIGMOID)
                        .build(),"GENCNN10")
        };
    }


    public static LEntry[] discriminatorLayers() {
        int channels = 3;
        double nonZeroBias = 1;

        return new LEntry[]{
                new LayerEntry("DISCNN1", new ConvolutionLayer.Builder(new int[]{11,11}, new int[]{4, 4})
                        .cudnnAlgoMode(ConvolutionLayer.AlgoMode.PREFER_FASTEST)
                        .convolutionMode(ConvolutionMode.Truncate)
                        .activation(Activation.RELU)
                        .nIn(channels)
                        .nOut(96)
                        .build(),
                        "Input"),
                new LayerEntry("DISLRN1", new LocalResponseNormalization.Builder().build(),"DISCNN1"),
                new LayerEntry("DISSL1", new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(3,3)
                        .stride(2,2)
                        .padding(1,1)
                        .build(),"DISLRN1"),
                new LayerEntry("DISCNN2", new ConvolutionLayer.Builder(new int[]{5,5}, new int[]{1,1}, new int[]{2,2})
                        .cudnnAlgoMode(ConvolutionLayer.AlgoMode.PREFER_FASTEST)
                        .convolutionMode(ConvolutionMode.Truncate)
                        .activation(Activation.RELU)
                        .nOut(256)
                        .biasInit(nonZeroBias)
                        .build(),"DISSL1"),
                new LayerEntry("DISSL2", new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{3, 3}, new int[]{2, 2})
                        .convolutionMode(ConvolutionMode.Truncate)
                        .build(),"DISCNN2"),
                new LayerEntry("DISLRN2", new LocalResponseNormalization.Builder().build(),"DISSL2"),
                new LayerEntry("DISCNN3", new ConvolutionLayer.Builder()
                        .kernelSize(3,3)
                        .stride(1,1)
                        .convolutionMode(ConvolutionMode.Same)
                        .activation(Activation.RELU)
                        .cudnnAlgoMode(ConvolutionLayer.AlgoMode.PREFER_FASTEST)
                        .nOut(384)
                        .build(),"DISLRN2"),
                new LayerEntry("DISCNN4", new ConvolutionLayer.Builder(new int[]{3,3}, new int[]{1,1})
                        .cudnnAlgoMode(ConvolutionLayer.AlgoMode.PREFER_FASTEST)
                        .activation(Activation.RELU)
                        .convolutionMode(ConvolutionMode.Same)
                        .nOut(384)
                        .biasInit(nonZeroBias)
                        .build(),"DISCNN3"),
                new LayerEntry("DISCNN5", new ConvolutionLayer.Builder(new int[]{3,3}, new int[]{1,1})
                        .cudnnAlgoMode(ConvolutionLayer.AlgoMode.PREFER_FASTEST)
                        .activation(Activation.RELU)
                        .convolutionMode(ConvolutionMode.Same)
                        .nOut(256)
                        .biasInit(nonZeroBias)
                        .build(),"DISCNN4"),
                new LayerEntry("DISSL3", new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{3,3}, new int[]{2,2})
                        .convolutionMode(ConvolutionMode.Truncate)
                        .build(),"DISCNN5"),
                new LayerEntry("DISFFN1", new DenseLayer.Builder()
                        .weightInit(new NormalDistribution(0, 0.005))
                        .activation(Activation.RELU)

                        .nOut(4096)
                        .biasInit(nonZeroBias)
                        .build(),"DISSL3"),
                new LayerEntry("DISFFN2", new DenseLayer.Builder()
                        .weightInit(new NormalDistribution(0, 0.005))
                        .activation(Activation.RELU)
                        .nOut(4096)
                        .biasInit(nonZeroBias)
                        .dropOut(0.5)
                        .build(),"DISFFN1"),
                new LayerEntry("DISLoss", new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(2)
                        .activation(Activation.SOFTMAX)
                        .weightInit(new NormalDistribution(0, 0.005))
                        .biasInit(0.1)
                        .build(),"DISFFN2")

        };
    }

    public static MultiLayerNetwork getDiscriminator() {
        NeuralNetConfiguration.ListBuilder builder = new NeuralNetConfiguration.Builder()
                .weightInit(new NormalDistribution(0.0,0.01))
                .updater(new Nesterovs(0.01,0.9))
                .biasUpdater(new Nesterovs(2e-2,0.9))
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .l2(5*1e-4)
                .miniBatch(false)

                .list()
                .setInputType(InputType.convolutional(256,256,3));

        //.addInputs("Input","Mask")
        //m + rgb 256x256x4x1
        //.setInputType(InputType.convolutional(256,256,3));

        for (int i = 0; i < discriminatorLayers().length; i++)
            builder.layer(i, ((LayerEntry)discriminatorLayers()[i]).getLayer());


        return new MultiLayerNetwork(builder.build());
    }
}