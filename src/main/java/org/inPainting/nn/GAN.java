package org.inPainting.nn;

import javafx.scene.image.WritableImage;
import lombok.Getter;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.api.BaseTrainingListener;
import org.inPainting.nn.entry.*;
import org.inPainting.nn.res.NetResult;
import org.inPainting.utils.ImageLoader;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.MultiDataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.*;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.function.Supplier;


public class GAN {
    private static final IUpdater UPDATER_ZERO = Sgd.builder().learningRate(0.00).build();

    public static final double LEARNING_RATE = 2E-4;
    public static final double LEARNING_BETA1 = 5E-1;
    public static final double LEARNING_LAMBDA = 10E+1;
    public static final int[] _MergedNetInputShape = {1,4,256,256};

    public interface DiscriminatorProvider {
        MultiLayerNetwork provide(IUpdater updater);
    }

    protected Supplier<ComputationGraph> generatorSupplier;
    protected DiscriminatorProvider discriminatorSupplier;

    @Getter
    protected MultiLayerNetwork discriminator;
    @Getter
    protected ComputationGraph network;

    protected IUpdater updater = new Adam(LEARNING_RATE);
    protected IUpdater biasUpdater;
    protected OptimizationAlgorithm optimizer;
    protected GradientNormalization gradientNormalizer;
    protected double gradientNormalizationThreshold;
    protected WorkspaceMode trainingWorkSpaceMode;
    protected WorkspaceMode inferenceWorkspaceMode;
    protected CacheMode cacheMode;
    protected long seed;
    protected ImageLoader imageLoader = new ImageLoader();



    public GAN(Builder builder) {
        this.generatorSupplier = builder.generator;
        this.discriminatorSupplier = builder.discriminator;
        this.updater = builder.iUpdater;
        this.biasUpdater = builder.biasUpdater;
        this.optimizer = builder.optimizationAlgo;
        this.gradientNormalizer = builder.gradientNormalization;
        this.gradientNormalizationThreshold = builder.gradientNormalizationThreshold;
        this.trainingWorkSpaceMode = builder.trainingWorkspaceMode;
        this.inferenceWorkspaceMode = builder.inferenceWorkspaceMode;
        this.cacheMode = builder.cacheMode;
        this.seed = builder.seed;

        this.defineGan();
    }

    public GAN(MultiLayerNetwork discriminator, ComputationGraph gan) {
        this.network = gan;
        this.discriminator = discriminator;
    }

    public WritableImage drawOutput(INDArray Picture, INDArray Mask, int width, int height) {
        return imageLoader.drawImage(network.output(Picture, Mask)[1], width, height);
    }

    public NetResult getOutput(INDArray Picture, INDArray Mask) {
        return new NetResult(network.output(Picture,Mask));
    }

    public Evaluation evaluateGan(DataSetIterator data) {
        return network.evaluate(data);
    }

    public Evaluation evaluateGan(DataSetIterator data, List<String> labelsList) {
        return network.evaluate(data, labelsList);
    }

    public void setDiscriminatorListeners(BaseTrainingListener[] listeners) {
        discriminator.setListeners(listeners);
    }

    public void setGanListeners(BaseTrainingListener[] listeners) {
        network.setListeners(listeners);
    }

    public void fit(MultiDataSetIterator realData, int numEpochs) {
        for (int i = 0; i < numEpochs; i++) {
            while (realData.hasNext()) {
                MultiDataSet next = (MultiDataSet) realData.next();
                fit(next,true);
            }
            realData.reset();
        }
    }

    public void fit(MultiDataSet next, boolean trainDiscriminator) {
        //INDArray realImage = next.getLabels()[0];
        /*for (int i = 0; i < discriminator.getLayers().length; i++) {
            if (discriminatorLearningRates[i] != null) {
                discriminator.setLearningRate(i, discriminatorLearningRates[i]);
            }
        }*/
        if (trainDiscriminator) {
            INDArray[] ganOutput = network.output(next.getFeatures());

            // Real images are marked as "0;1", fake images at "1;0".
            //DataSet realSet = new DataSet(next.getLabels()[0], Outputs.REAL());
            //DataSet fakeSet = new DataSet(ganOutput[1], Outputs.FAKE());

            MultiDataSet fakeSetOutput = new MultiDataSet(
                    new INDArray[]{
                            ganOutput[1], //gan output
                            //next.getFeatures()[1] //mask
                    },new INDArray[]{
                    Outputs.FAKE()
            });

            MultiDataSet realSet = new MultiDataSet(
                    new INDArray[]{
                            next.getLabels()[0], //expected output
                            //next.getFeatures()[1] //mask
                    },new INDArray[]{
                    Outputs.REAL()
            });

            /*
            MultiDataSet fakeSetInput = new MultiDataSet(
                    new INDArray[]{
                            next.getFeatures()[0], //input
                            next.getFeatures()[1] //mask
                    },new INDArray[]{
                    Outputs.FAKE()
            });
             */

            discriminator.fit(fakeSetOutput);
            discriminator.fit(realSet);

            //discriminator.fit(fakeSetInput);
            //discriminator.fit(realSet);

            fakeSetOutput.detach();
            realSet.detach();
            //fakeSetInput.detach();
        }

        // Generate a new set of adversarial examples and try to mislead the discriminator.
        // by labeling the fake images as real images we reward the generator when it's output
        // tricks the discriminator.
        //INDArray adversarialExamples = Nd4j.rand(new int[]{batchSize, latentDim});
        //INDArray misleadingLabels = Nd4j.zeros(batchSize, 1);

        //DataSet adversarialSet = new DataSet(next.getFeatures(), realone);
        // Set learning rate of discriminator part of gan to zero.
        /*for (int i = generator.getLayers().length; i < gan.getLayers().length; i++) {
            gan.setLearningRate(i, 0.0);
        }*/

        // Fit the Pix2PixGAN on the adversarial set, trying to fool the discriminator by generating
        // better fake images.
        network.fit(new MultiDataSet(
                new INDArray[]{
                        next.getFeatures()[0],
                        next.getFeatures()[1]
                },
                new INDArray[]{
                        Outputs.REAL(),
                        next.getLabels()[0]
                })
        );

        // Update the discriminator in the Pix2PixGAN network
        updateGanWithDiscriminator();
    }

    private void defineGan() {
        discriminator = discriminatorSupplier.provide(updater);
        discriminator.init();

        MultiLayerNetwork ganDiscriminator = discriminatorSupplier.provide(UPDATER_ZERO);
        ganDiscriminator.init();

        network = NET(updater);
        network.init();

        // Update the discriminator in the Pix2PixGAN network
        updateGanWithDiscriminator();
    }

    public ComputationGraph NET(IUpdater updater) {
        int outputChannels = 3;
        int maskChannels = 1;

        ComputationGraphConfiguration.GraphBuilder graphBuilder = new NeuralNetConfiguration.Builder()
                .updater(updater)
                //.l1(5* 1E-4)
                .graphBuilder()
                .allowDisconnected(true)
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

        //Generator layers
        LEntry[] GenlEntry = NeuralNetwork.genLayers();
        for (int i = 0; i < GenlEntry.length; i++) {
            if (!GenlEntry[i].isVertex())
                graphBuilder.addLayer(
                        GenlEntry[i].getLayerName(),
                        ((LayerEntry)GenlEntry[i]).getLayer(),
                        GenlEntry[i].getInputs()
                );
            else
                graphBuilder.addVertex(
                        GenlEntry[i].getLayerName(),
                        ((VertexEntry)GenlEntry[i]).getVertex(),
                        GenlEntry[i].getInputs()
                );
        }


        //Discriminator layers
        LEntry[] DislEntry = NeuralNetwork.discriminatorLayers();

        //Changing first inputs os first layer of discriminator
        graphBuilder.addLayer(
                DislEntry[0].getLayerName(),
                ((LayerEntry)DislEntry[0]).getLayer(),
                GenlEntry[GenlEntry.length - 2].getLayerName());

        for (int i = 1; i < DislEntry.length; i++) {

            graphBuilder.addLayer(
                    DislEntry[i].getLayerName(),
                    ((LayerEntry) DislEntry[i]).getLayer(),
                    DislEntry[i].getInputs()
            );
        }
        graphBuilder.setOutputs("DISLoss","GENCNNLoss"); //Discriminator output, Generator loss

        return new ComputationGraph(graphBuilder.build());
    }

    /**
     * After the discriminator has been trained, we update the respective parts of the GAN network
     * as well.
     */
    private void updateGanWithDiscriminator() {
        int genLayerCount = network.getLayers().length - discriminator.getLayers().length; //Position of first Discriminator Layer
        for (int i = genLayerCount; i < network.getLayers().length; i++) {
            network.getLayer(i).setParams(discriminator.getLayer(i - genLayerCount).params());
            network.getLayer(i).setMaskArray(discriminator.getLayer(i-genLayerCount).getMaskArray());
        }
    }

    /**
     * GAN builder, used as a starting point for creating a MultiLayerConfiguration or
     * ComputationGraphConfiguration.<br>
     */
    public static class Builder implements Cloneable {
        protected Supplier<ComputationGraph> generator;
        protected DiscriminatorProvider discriminator;

        protected IUpdater iUpdater = new Sgd();
        protected IUpdater biasUpdater = null;
        protected long seed = System.currentTimeMillis();
        protected OptimizationAlgorithm optimizationAlgo = OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT;
        protected GradientNormalization gradientNormalization = GradientNormalization.None;
        protected double gradientNormalizationThreshold = 1.0;

        protected WorkspaceMode trainingWorkspaceMode = WorkspaceMode.ENABLED;
        protected WorkspaceMode inferenceWorkspaceMode = WorkspaceMode.ENABLED;
        protected CacheMode cacheMode = CacheMode.NONE;

        public Builder() {
        }

        /**
         * Set the image discriminator of the GAN.
         *
         * @param discriminator MultilayerNetwork
         * @return Builder
         */
        public GAN.Builder discriminator(DiscriminatorProvider discriminator) {
            this.discriminator = discriminator;
            return this;
        }

        /**
         * Random number generator seed. Used for reproducibility between runs
         */
        public GAN.Builder seed(long seed) {
            this.seed = seed;
            Nd4j.getRandom().setSeed(seed);
            return this;
        }

        /**
         * Optimization algorithm to use. Most common: OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT
         *
         * @param optimizationAlgo Optimization algorithm to use when training
         */
        public GAN.Builder optimizationAlgo(OptimizationAlgorithm optimizationAlgo) {
            this.optimizationAlgo = optimizationAlgo;
            return this;
        }


        /**
         * Gradient updater configuration. For example, {@link org.nd4j.linalg.learning.config.Adam}
         * or {@link org.nd4j.linalg.learning.config.Nesterovs}<br>
         * Note: values set by this method will be applied to all applicable layers in the network, unless a different
         * value is explicitly set on a given layer. In other words: values set via this method are used as the default
         * value, and can be overridden on a per-layer basis.
         *
         * @param updater Updater to use
         */
        public GAN.Builder updater(IUpdater updater) {
            this.iUpdater = updater;
            return this;
        }

        /**
         * Gradient updater configuration, for the biases only. If not set, biases will use the updater as
         * set by {@link #updater(IUpdater)}<br>
         * Note: values set by this method will be applied to all applicable layers in the network, unless a different
         * value is explicitly set on a given layer. In other words: values set via this method are used as the default
         * value, and can be overridden on a per-layer basis.
         *
         * @param updater Updater to use for bias parameters
         */
        public GAN.Builder biasUpdater(IUpdater updater) {
            this.biasUpdater = updater;
            return this;
        }

        /**
         * Gradient normalization strategy. Used to specify gradient renormalization, gradient clipping etc.
         * See {@link GradientNormalization} for details<br>
         * Note: values set by this method will be applied to all applicable layers in the network, unless a different
         * value is explicitly set on a given layer. In other words: values set via this method are used as the default
         * value, and can be overridden on a per-layer basis.
         *
         * @param gradientNormalization Type of normalization to use. Defaults to None.
         * @see GradientNormalization
         */
        public GAN.Builder gradientNormalization(GradientNormalization gradientNormalization) {
            this.gradientNormalization = gradientNormalization;
            return this;
        }

        /**
         * Threshold for gradient normalization, only used for GradientNormalization.ClipL2PerLayer,
         * GradientNormalization.ClipL2PerParamType, and GradientNormalization.ClipElementWiseAbsoluteValue<br>
         * Not used otherwise.<br>
         * L2 threshold for first two types of clipping, or absolute value threshold for last type of clipping.<br>
         * Note: values set by this method will be applied to all applicable layers in the network, unless a different
         * value is explicitly set on a given layer. In other words: values set via this method are used as the default
         * value, and can be overridden on a per-layer basis.
         */
        public GAN.Builder gradientNormalizationThreshold(double threshold) {
            this.gradientNormalizationThreshold = threshold;
            return this;
        }

        public GAN build() {
            return new GAN(this);
        }
    }

    public static class Outputs {
        public static INDArray REAL(){
            INDArray real_one = Nd4j.zeros(1,2);
            for(int i = 0; i<2; i++)
                real_one.putScalar(i,i);

            return real_one;
        }

        public static INDArray FAKE(){
            INDArray fake_one = Nd4j.zeros(1,2);
            fake_one.putScalar(0, 1);
            fake_one.putScalar(1, 0);

            return fake_one;
        }
    }
}