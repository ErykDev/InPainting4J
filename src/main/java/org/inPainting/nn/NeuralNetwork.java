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
import org.deeplearning4j.nn.conf.layers.misc.FrozenLayerWithBackprop;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ops.impl.scalar.LeakyReLU;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.inPainting.nn.entry.LEntry;
import org.inPainting.nn.entry.LayerEntry;
import org.inPainting.nn.entry.VertexEntry;

import java.io.File;
import java.io.IOException;

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

        ConvolutionLayer.AlgoMode cudnnAlgoMode = ConvolutionLayer.AlgoMode.PREFER_FASTEST;

        return new LEntry[]{
                //new VertexEntry("merge1", new MergeVertex(), "Input","Mask"),

                new LayerEntry("conv1-1", new ConvolutionLayer.Builder(3,3).stride(1,1).nOut(64)
                        .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(cudnnAlgoMode)
                        .activation(Activation.RELU).build(), "Input"),
                new LayerEntry("conv1-2", new ConvolutionLayer.Builder(3,3).stride(1,1).nOut(64)
                        .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(cudnnAlgoMode)
                        .activation(Activation.RELU).build(), "conv1-1"),
                new LayerEntry("pool1", new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2,2)
                        .build(), "conv1-2"),

                new LayerEntry("conv2-1", new ConvolutionLayer.Builder(3,3).stride(1,1).nOut(128)
                        .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(cudnnAlgoMode)
                        .activation(Activation.RELU).build(), "pool1"),
                new LayerEntry("conv2-2", new ConvolutionLayer.Builder(3,3).stride(1,1).nOut(128)
                        .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(cudnnAlgoMode)
                        .activation(Activation.RELU).build(), "conv2-1"),
                new LayerEntry("pool2", new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2,2)
                        .build(), "conv2-2"),

                new LayerEntry("conv3-1", new ConvolutionLayer.Builder(3,3).stride(1,1).nOut(256)
                        .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(cudnnAlgoMode)
                        .activation(Activation.RELU).build(), "pool2"),
                new LayerEntry("conv3-2", new ConvolutionLayer.Builder(3,3).stride(1,1).nOut(256)
                        .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(cudnnAlgoMode)
                        .activation(Activation.RELU).build(), "conv3-1"),
                new LayerEntry("pool3", new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2,2)
                        .build(), "conv3-2"),

                new LayerEntry("conv4-1", new ConvolutionLayer.Builder(3,3).stride(1,1).nOut(512)
                        .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(cudnnAlgoMode)
                        .activation(Activation.RELU).build(), "pool3"),
                new LayerEntry("conv4-2", new ConvolutionLayer.Builder(3,3).stride(1,1).nOut(512)
                        .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(cudnnAlgoMode)
                        .activation(Activation.RELU).build(), "conv4-1"),
                new LayerEntry("drop4", new DropoutLayer.Builder(0.5).build(), "conv4-2"),
                new LayerEntry("pool4", new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2,2)
                        .build(), "drop4"),

                new LayerEntry("conv5-1", new ConvolutionLayer.Builder(3,3).stride(1,1).nOut(1024)
                        .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(cudnnAlgoMode)
                        .activation(Activation.RELU).build(), "pool4"),
                new LayerEntry("conv5-2", new ConvolutionLayer.Builder(3,3).stride(1,1).nOut(1024)
                        .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(cudnnAlgoMode)
                        .activation(Activation.RELU).build(), "conv5-1"),
                new LayerEntry("drop5", new DropoutLayer.Builder(0.5).build(), "conv5-2"),

                // up6
                new LayerEntry("up6-1", new Upsampling2D.Builder(2).build(), "drop5"),
                new LayerEntry("up6-2", new ConvolutionLayer.Builder(2,2).stride(1,1).nOut(512)
                        .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(cudnnAlgoMode)
                        .activation(Activation.RELU).build(), "up6-1"),
                new VertexEntry("merge6", new MergeVertex(), "drop4", "up6-2"),
                new LayerEntry("conv6-1", new ConvolutionLayer.Builder(3,3).stride(1,1).nOut(512)
                        .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(cudnnAlgoMode)
                        .activation(Activation.RELU).build(), "merge6"),
                new LayerEntry("conv6-2", new ConvolutionLayer.Builder(3,3).stride(1,1).nOut(512)
                        .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(cudnnAlgoMode)
                        .activation(Activation.RELU).build(), "conv6-1"),

                // up7
                new LayerEntry("up7-1", new Upsampling2D.Builder(2).build(), "conv6-2"),
                new LayerEntry("up7-2", new ConvolutionLayer.Builder(2,2).stride(1,1).nOut(256)
                        .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(cudnnAlgoMode)
                        .activation(Activation.RELU).build(), "up7-1"),
                new VertexEntry("merge7", new MergeVertex(), "conv3-2", "up7-2"),
                new LayerEntry("conv7-1", new ConvolutionLayer.Builder(3,3).stride(1,1).nOut(256)
                        .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(cudnnAlgoMode)
                        .activation(Activation.RELU).build(), "merge7"),
                new LayerEntry("conv7-2", new ConvolutionLayer.Builder(3,3).stride(1,1).nOut(256)
                        .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(cudnnAlgoMode)
                        .activation(Activation.RELU).build(), "conv7-1"),

                // up8
                new LayerEntry("up8-1", new Upsampling2D.Builder(2).build(), "conv7-2"),
                new LayerEntry("up8-2", new ConvolutionLayer.Builder(2,2).stride(1,1).nOut(128)
                        .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(cudnnAlgoMode)
                        .activation(Activation.RELU).build(), "up8-1"),
                new VertexEntry("merge8", new MergeVertex(), "conv2-2", "up8-2"),
                new LayerEntry("conv8-1", new ConvolutionLayer.Builder(3,3).stride(1,1).nOut(128)
                        .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(cudnnAlgoMode)
                        .activation(Activation.RELU).build(), "merge8"),
                new LayerEntry("conv8-2", new ConvolutionLayer.Builder(3,3).stride(1,1).nOut(128)
                        .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(cudnnAlgoMode)
                        .activation(Activation.RELU).build(), "conv8-1"),

                // up9
                new LayerEntry("up9-1", new Upsampling2D.Builder(2).build(), "conv8-2"),
                new LayerEntry("up9-2", new ConvolutionLayer.Builder(2,2).stride(1,1).nOut(64)
                        .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(cudnnAlgoMode)
                        .activation(Activation.RELU).build(), "up9-1"),
                new VertexEntry("merge9", new MergeVertex(), "conv1-2", "up9-2"),
                new LayerEntry("conv9-1", new ConvolutionLayer.Builder(3,3).stride(1,1).nOut(64)
                        .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(cudnnAlgoMode)
                        .activation(Activation.RELU).build(), "merge9"),
                new LayerEntry("conv9-2", new ConvolutionLayer.Builder(3,3).stride(1,1).nOut(64)
                        .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(cudnnAlgoMode)
                        .activation(Activation.RELU).build(), "conv9-1"),
                new LayerEntry("conv9-3", new ConvolutionLayer.Builder(3,3).stride(1,1).nOut(64)
                        .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(cudnnAlgoMode)
                        .activation(Activation.RELU).build(), "conv9-2"),

                new LayerEntry("conv10", new ConvolutionLayer.Builder(1,1).stride(1,1).nOut(3)
                        .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(cudnnAlgoMode)
                        .activation(Activation.IDENTITY).build(), "conv9-3"),
                new LayerEntry("GENCNNLoss", new CnnLossLayer.Builder(LossFunctions.LossFunction.MEAN_ABSOLUTE_ERROR)
                        .activation(Activation.SIGMOID).build(), "conv10")
        };
    }

    public static LEntry[] discriminatorLayers() {
        int channels = 6;
        ConvolutionLayer.AlgoMode cudnnAlgoMode = ConvolutionLayer.AlgoMode.PREFER_FASTEST;

        return new LEntry[]{
                new VertexEntry("merge2", new MergeVertex(), "Input1","Input2"),

                // #C64
                new LayerEntry("conv11", new ConvolutionLayer.Builder(new int[]{4,4}, new int[]{2,2})
                        .cudnnAlgoMode(cudnnAlgoMode).convolutionMode(ConvolutionMode.Same)
                        .nIn(channels).nOut(64)
                        .build(), "merge2"),
                new LayerEntry("al0", new ActivationLayer.Builder()
                        .activation(Activation.LEAKYRELU).build(),"conv11"),

                // #C128
                new LayerEntry("conv12", new ConvolutionLayer.Builder(new int[]{4,4}, new int[]{2,2})
                        .cudnnAlgoMode(cudnnAlgoMode).convolutionMode(ConvolutionMode.Same)
                        .nOut(128)
                        .build(),"al0"),
                new LayerEntry("lrn1", new LocalResponseNormalization.Builder().build(),"conv12"),
                new LayerEntry("al1", new ActivationLayer.Builder()
                        .activation(Activation.LEAKYRELU).build(),"lrn1"),

                // #C256
                new LayerEntry("conv13", new ConvolutionLayer.Builder(new int[]{4,4}, new int[]{2,2})
                        .cudnnAlgoMode(cudnnAlgoMode).convolutionMode(ConvolutionMode.Same)
                        .nOut(256)
                        .build(),"al1"),
                new LayerEntry("lrn2", new LocalResponseNormalization.Builder().build(),"conv13"),
                new LayerEntry("al2", new ActivationLayer.Builder()
                        .activation(Activation.LEAKYRELU).build(),"lrn2"),

                // #C512
                new LayerEntry("conv14", new ConvolutionLayer.Builder(new int[]{4,4}, new int[]{2,2})
                        .cudnnAlgoMode(cudnnAlgoMode).convolutionMode(ConvolutionMode.Same)
                        .nOut(512)
                        .build(),"al2"),
                new LayerEntry("lrn3", new LocalResponseNormalization.Builder().build(),"conv14"),
                new LayerEntry("al3", new ActivationLayer.Builder()
                        .activation(Activation.LEAKYRELU).build(),"lrn3"),

                // #second last output layer
                new LayerEntry("conv15", new ConvolutionLayer.Builder(new int[]{4,4})
                        .cudnnAlgoMode(cudnnAlgoMode).convolutionMode(ConvolutionMode.Same)
                        .nOut(512)
                        .build(),"al3"),
                new LayerEntry("lrn4", new LocalResponseNormalization.Builder().build(),"conv15"),
                new LayerEntry("al4", new ActivationLayer.Builder()
                        .activation(Activation.LEAKYRELU).build(),"lrn4"),

                // #patch output
                new LayerEntry("conv16", new ConvolutionLayer.Builder(new int[]{4,4})
                        .cudnnAlgoMode(cudnnAlgoMode).convolutionMode(ConvolutionMode.Same)
                        .nOut(1)
                        .build(),"al4"),

                new LayerEntry("DISLoss", new CnnLossLayer.Builder(LossFunctions.LossFunction.XENT)
                        .activation(Activation.SIGMOID).build(), "conv16")
        };
    }


    public static ComputationGraph getDiscriminator() {
        int[] imageInputShape = {1,3,256,256};

        ComputationGraphConfiguration.GraphBuilder graphBuilder = new NeuralNetConfiguration.Builder()
                .updater(Adam.builder()
                        .learningRate(0.0004)
                        .beta1(0.5)
                        .beta2(0.999)
                        .build())

                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .weightInit(new NormalDistribution(0.0,0.02))

                .graphBuilder()

                .addInputs("Input1", "Input2")
                //rgb 256x256x3x1 + 256x256x3x1
                .setInputTypes(InputType.convolutional(
                        imageInputShape[2],
                        imageInputShape[3],
                        imageInputShape[1]
                ), InputType.convolutional(
                        imageInputShape[2],
                        imageInputShape[3],
                        imageInputShape[1]
                ));

        for (int i = 0; i < discriminatorLayers().length; i++)
            if (!discriminatorLayers()[i].isVertex())
                graphBuilder.addLayer(discriminatorLayers()[i].getLayerName(), ((LayerEntry)discriminatorLayers()[i]).getLayer(), ((LayerEntry)discriminatorLayers()[i]).getInputs());
            else
                graphBuilder.addVertex(discriminatorLayers()[i].getLayerName(), ((VertexEntry)discriminatorLayers()[i]).getVertex(), ((VertexEntry)discriminatorLayers()[i]).getInputs());

        graphBuilder.setOutputs("DISLoss");

        return new ComputationGraph(graphBuilder.build());
    }
}