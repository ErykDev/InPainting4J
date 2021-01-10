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
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
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
                new VertexEntry("merge1", new MergeVertex(), "Input","Mask"),

                new LayerEntry("conv1-1", new ConvolutionLayer.Builder(3,3).stride(1,1).nOut(64)
                        .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(cudnnAlgoMode)
                        .activation(Activation.RELU).build(), "merge1"),
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
                new LayerEntry("GENCNNLoss", new CnnLossLayer.Builder(LossFunctions.LossFunction.XENT)
                        .activation(Activation.SIGMOID).build(), "conv10")


        };
    }

    public static LEntry[] discriminatorLayers() {
        int channels = 4;

        return new LEntry[]{
                new VertexEntry("merge2", new MergeVertex(), "Input","Mask"),

                // #C64
                new LayerEntry("DISCNN1", new ConvolutionLayer.Builder(new int[]{4,4}, new int[]{2,2})
                        .cudnnAlgoMode(ConvolutionLayer.AlgoMode.PREFER_FASTEST)
                        .convolutionMode(ConvolutionMode.Same)
                        .activation(Activation.LEAKYRELU)
                        .nIn(channels)
                        .nOut(64)
                        .build(),
                        "merge2"),

                // #C128
                new LayerEntry("DISCNN2", new ConvolutionLayer.Builder(new int[]{4,4}, new int[]{2,2})
                        .cudnnAlgoMode(ConvolutionLayer.AlgoMode.PREFER_FASTEST)
                        .convolutionMode(ConvolutionMode.Same)
                        .activation(Activation.LEAKYRELU)
                        .nOut(128)
                        .build(),"DISCNN1"),
                new LayerEntry("DISLRN2", new LocalResponseNormalization.Builder().build(),"DISCNN2"),

                // #C256
                new LayerEntry("DISCNN3", new ConvolutionLayer.Builder(new int[]{4,4}, new int[]{2,2})
                        .cudnnAlgoMode(ConvolutionLayer.AlgoMode.PREFER_FASTEST)
                        .convolutionMode(ConvolutionMode.Same)
                        .activation(Activation.LEAKYRELU)
                        .nOut(256)
                        .build(),"DISLRN2"),
                new LayerEntry("DISLRN3", new LocalResponseNormalization.Builder().build(),"DISCNN3"),

                // #C512
                new LayerEntry("DISCNN4", new ConvolutionLayer.Builder(new int[]{4,4}, new int[]{2,2})
                        .cudnnAlgoMode(ConvolutionLayer.AlgoMode.PREFER_FASTEST)
                        .convolutionMode(ConvolutionMode.Same)
                        .activation(Activation.LEAKYRELU)
                        .nOut(512)
                        .build(),"DISLRN3"),
                new LayerEntry("DISLRN4", new LocalResponseNormalization.Builder().build(),"DISCNN4"),


                // #second last output layer
                new LayerEntry("DISCNN5", new ConvolutionLayer.Builder(new int[]{4,4})
                        .cudnnAlgoMode(ConvolutionLayer.AlgoMode.PREFER_FASTEST)
                        .convolutionMode(ConvolutionMode.Same)
                        .activation(Activation.LEAKYRELU)
                        .nOut(512)
                        .build(),"DISLRN4"),
                new LayerEntry("DISLRN5", new LocalResponseNormalization.Builder().build(),"DISCNN5"),


                // #patch output
                new LayerEntry("DISCNN6", new ConvolutionLayer.Builder(new int[]{4,4})
                        .cudnnAlgoMode(ConvolutionLayer.AlgoMode.PREFER_FASTEST)
                        .convolutionMode(ConvolutionMode.Same)
                        .activation(Activation.LEAKYRELU)
                        .nOut(1)
                        .build(),"DISLRN5"),


                new LayerEntry("DISFFN1", new DenseLayer.Builder()
                        .weightInit(new NormalDistribution(0, 0.005))
                        .activation(Activation.LEAKYRELU)
                        .nOut(4096)
                        .build(),"DISCNN6"),
                new LayerEntry("DISFFN2", new DenseLayer.Builder()
                        .weightInit(new NormalDistribution(0, 0.005))
                        .activation(Activation.LEAKYRELU)
                        .nOut(4096)
                        .dropOut(0.5)
                        .build(),"DISFFN1"),

                new LayerEntry("DISLoss", new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .weightInit(new NormalDistribution(0, 0.005))
                        .activation(Activation.SOFTMAX)
                        .biasInit(0.1)
                        .nOut(2)
                        .build(),"DISFFN2")
        };
    }


    public static ComputationGraph getDiscriminator() {
        int[] _MergedNetInputShape = {1,4,256,256};
        int outputChannels = 3;
        int maskChannels = 1;

        ComputationGraphConfiguration.GraphBuilder graphBuilder = new NeuralNetConfiguration.Builder()
                .weightInit(new NormalDistribution(0.0,0.02))
                .updater(Adam.builder()
                        .learningRate(0.0002)
                        .beta1(0.5)
                        .beta2(0.999)
                        .build())
                //.biasUpdater(new Nesterovs(2e-2,0.9))
                //.gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                //.l2(5*1e-4)
                .miniBatch(false)


                .graphBuilder()

                .addInputs("Input", "Mask")
                //rgb 256x256x3x1 + m 256x256x1x1
                .setInputTypes(InputType.convolutional(
                        _MergedNetInputShape[2],
                        _MergedNetInputShape[3],
                        _MergedNetInputShape[1] - maskChannels
                ), InputType.convolutional(
                        _MergedNetInputShape[2],
                        _MergedNetInputShape[3],
                        _MergedNetInputShape[1] - outputChannels
                ));

        //m + rgb 256x256x4x1
        //.setInputType(InputType.convolutional(256,256,3));
        for (int i = 0; i < discriminatorLayers().length; i++)
            if (!discriminatorLayers()[i].isVertex())
                graphBuilder.addLayer(((LayerEntry)discriminatorLayers()[i]).getLayerName(), ((LayerEntry)discriminatorLayers()[i]).getLayer(), ((LayerEntry)discriminatorLayers()[i]).getInputs());
            else
                graphBuilder.addVertex(((VertexEntry)discriminatorLayers()[i]).getLayerName(), ((VertexEntry)discriminatorLayers()[i]).getVertex(), ((VertexEntry)discriminatorLayers()[i]).getInputs());

        graphBuilder.setOutputs("DISLoss");

        return new ComputationGraph(graphBuilder.build());
    }
}