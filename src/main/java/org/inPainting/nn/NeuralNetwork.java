package org.inPainting.nn;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.inPainting.nn.entry.LEntry;
import org.inPainting.nn.entry.LayerEntry;
import org.inPainting.nn.entry.VertexEntry;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class NeuralNetwork {

    /**
     * @return predetermined layers of U-net Generator
     */
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
                new LayerEntry("GENCNNLoss", new CnnLossLayer.Builder(LossFunctions.LossFunction.MEAN_ABSOLUTE_ERROR)
                        .activation(Activation.SIGMOID).build(), "conv10")
        };
    }


    /**
     * @return predetermined layers of 70x70 PatchGan as discriminator (C64-C128-C256-C512)
     */
    public static LEntry[] discriminatorLayers() {
        int channels = 6+1; //two images + mask
        ConvolutionLayer.AlgoMode cudnnAlgoMode = ConvolutionLayer.AlgoMode.PREFER_FASTEST;

        return new LEntry[]{
                new VertexEntry("merge2", new MergeVertex(), "Input1", "Input2", "Mask"),

                // #C64
                new LayerEntry("conv11", new ConvolutionLayer.Builder(4,4).stride(2,2)
                        .cudnnAlgoMode(cudnnAlgoMode).convolutionMode(ConvolutionMode.Same)
                        .activation(Activation.LEAKYRELU)
                        .nIn(channels).nOut(64)
                        .build(), "merge2"),
                new LayerEntry("lrn1", new BatchNormalization.Builder().build(),"conv11"),

                // #C128
                new LayerEntry("conv12", new ConvolutionLayer.Builder(4,4).stride(2,2)
                        .cudnnAlgoMode(cudnnAlgoMode).convolutionMode(ConvolutionMode.Same)
                        .activation(Activation.LEAKYRELU)
                        .nOut(128)
                        .build(),"lrn1"),
                new LayerEntry("lrn2", new BatchNormalization.Builder().build(),"conv12"),

                // #C256
                new LayerEntry("conv13", new ConvolutionLayer.Builder(4,4).stride(2,2)
                        .cudnnAlgoMode(cudnnAlgoMode).convolutionMode(ConvolutionMode.Same)
                        .activation(Activation.LEAKYRELU)
                        .nOut(128*2)
                        .build(),"lrn2"),
                new LayerEntry("lrn3", new BatchNormalization.Builder().build(),"conv13"),

                // #C512
                new LayerEntry("conv14", new ConvolutionLayer.Builder(4,4).stride(2,2)
                        .cudnnAlgoMode(cudnnAlgoMode).convolutionMode(ConvolutionMode.Same)
                        .activation(Activation.LEAKYRELU)
                        .nOut(128*4)
                        .build(),"lrn3"),
                new LayerEntry("lrn4", new BatchNormalization.Builder().build(),"conv14"),

                // #second last output layer
                new LayerEntry("conv15", new ConvolutionLayer.Builder(4,4)
                        .cudnnAlgoMode(cudnnAlgoMode).convolutionMode(ConvolutionMode.Same)
                        .activation(Activation.LEAKYRELU)
                        .nOut(128*4)
                        .build(),"lrn4"),
                new LayerEntry("lrn5", new BatchNormalization.Builder().build(),"conv15"),

                // #patch output
                new LayerEntry("conv16", new ConvolutionLayer.Builder(4,4)
                        .cudnnAlgoMode(cudnnAlgoMode).convolutionMode(ConvolutionMode.Same)
                        .activation(Activation.LEAKYRELU)
                        .nOut(1)
                        .build(),"lrn5"),

                new LayerEntry("DISLoss", new CnnLossLayer.Builder(LossFunctions.LossFunction.XENT)
                        .activation(Activation.SIGMOID).build(), "conv16")
        };
    }


    /**
     * Builds Discriminator network from discriminatorLayers() layers
     * @see LEntry[] discriminatorLayers()
     *
     * @return Initialized Discriminator network
     */
    public static ComputationGraph getDiscriminator() {
        InputType rgbImage = InputType.convolutional(256, 256, 3);
        InputType mask = InputType.convolutional(256, 256, 1);

        ComputationGraphConfiguration.GraphBuilder graphBuilder = new NeuralNetConfiguration.Builder()
                .weightInit(new NormalDistribution(0.0, 0.02))
                //.weightInit(WeightInit.RELU)

                .updater(Adam.builder()
                        .learningRate(0.0002)
                        .beta1(0.5)
                        .build())

                //.l1(5 * 1e-4)
                //.gradientNormalization(GradientNormalization.RenormalizeL2PerLayer) // normalize to prevent vanishing or exploding gradients
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)

                .graphBuilder()

                .addInputs("Input1", "Input2", "Mask")
                //rgb 256x256x3x1 + 256x256x3x1 + 256x256x1x1
                .setInputTypes(rgbImage,rgbImage, mask);


        for (LEntry entry: discriminatorLayers()) {
            if (entry.isVertex())
                graphBuilder.addVertex(entry.getLayerName(), ((VertexEntry)entry).getVertex(), entry.getInputs());
            else
                graphBuilder.addLayer(entry.getLayerName(), ((LayerEntry)entry).getLayer(), entry.getInputs());
        }

        graphBuilder.setOutputs("DISLoss");

        return new ComputationGraph(graphBuilder.build());
    }
}