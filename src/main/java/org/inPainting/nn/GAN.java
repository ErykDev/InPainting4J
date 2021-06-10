package org.inPainting.nn;

import javafx.scene.image.WritableImage;
import lombok.Getter;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.BaseTrainingListener;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.MultiDataSet;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.learning.config.Sgd;
import org.inPainting.nn.entry.LEntry;
import org.inPainting.nn.entry.LayerEntry;
import org.inPainting.nn.entry.VertexEntry;
import org.inPainting.nn.res.NetResult;
import org.inPainting.utils.ImageLoader;

import java.util.function.Supplier;

public class GAN {

    public static final double LEARNING_RATE = 0.0002;
    public static final double LEARNING_BETA1 = 0.5;
    public static final double LEARNING_LAMBDA = 100;
    public static final int[][] _InputShape = {
            {1,3,256,256},
            {1,1,256,256}
    };


    protected Supplier<ComputationGraph> generatorSupplier;
    protected Supplier<ComputationGraph> discriminatorSupplier;

    @Getter
    protected ComputationGraph discriminator;
    @Getter
    protected ComputationGraph network;

    protected IUpdater updater = new Sgd(LEARNING_RATE);
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

    public GAN(ComputationGraph discriminator, ComputationGraph gan) {
        this.network = gan;
        this.discriminator = discriminator;
    }

    public WritableImage drawOutput(INDArray Picture, INDArray Mask, int width, int height) {
        return imageLoader.drawImage(network.output(Picture, Mask)[1], width, height);
    }

    public NetResult getOutput(INDArray[] Picture) {
        return new NetResult(network.output(Picture));
    }

    public Evaluation evaluateGan(MultiDataSetIterator data) {
        return network.evaluate(data);
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
        if (trainDiscriminator) {
            INDArray[] ganOutput = network.output(next.getFeatures());

            // treating input as Fake
            MultiDataSet inputSet = new MultiDataSet(
                    new INDArray[] {
                            next.getFeatures()[0], //input
                            next.getFeatures()[0], //input
                    }, new INDArray[] {
                    Outputs.FAKE //zeros
            });

            // Fake images are marked as "0".
            MultiDataSet fakeSetOutput = new MultiDataSet(
                    new INDArray[]{
                            ganOutput[1], //gan output
                            next.getFeatures()[0], //input
                    },new INDArray[] {
                    Outputs.FAKE //zeros
            });

            // Real images are marked as "1"
            MultiDataSet realSet = new MultiDataSet(
                    new INDArray[]{
                            next.getLabels()[0], //expected output
                            next.getFeatures()[0], //input
                    },new INDArray[] {
                    Outputs.REAL //ones
            });

            discriminator.fit(inputSet);
            //for (int i = 0; i < 2; i++)
            discriminator.fit(realSet);

            discriminator.fit(fakeSetOutput);


            for (INDArray indArray: ganOutput)
                indArray = null;
            ganOutput = null;


            // Update the discriminator in the Pix2PixGAN network
            updateGanWithDiscriminator();
        }

        // Fit the Pix2PixGAN on the adversarial set, trying to fool the discriminator by generating
        // better fake images.
        network.fit(new MultiDataSet(
                next.getFeatures(), // Image And Mask

                new INDArray[] {
                        Outputs.REAL,
                        next.getLabels()[0]
                })
        );
    }

    private void defineGan() {
        discriminator = discriminatorSupplier.get();
        discriminator.init();

        ComputationGraph ganDiscriminator = discriminatorSupplier.get();
        ganDiscriminator.init();

        network = NET(updater);
        network.init();

        // Update the discriminator in the Pix2PixGAN network
        updateGanWithDiscriminator();
    }

    public ComputationGraph NET(IUpdater updater) {
        ComputationGraphConfiguration.GraphBuilder graphBuilder = new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .weightInit(WeightInit.RELU)
                .updater(updater)
                .l2(5e-5)
                .miniBatch(true)
                .graphBuilder()
                .addInputs("Input")
                //rgb 256x256x3x1 + m 256x256x1x1
                .setInputTypes(InputType.convolutional(
                        _InputShape[0][3],
                        _InputShape[0][2],
                        _InputShape[0][1]
                ));

        //Generator layers
        LEntry[] GenlEntry = NeuralNetwork.genLayers();
        for (LEntry lEntry : GenlEntry) {
            if (lEntry.isVertex())
                graphBuilder.addVertex(
                        lEntry.getLayerName(),
                        ((VertexEntry) lEntry).getVertex(),
                        lEntry.getInputs());
            else
                graphBuilder.addLayer(
                        lEntry.getLayerName(),
                        ((LayerEntry) lEntry).getLayer(),
                        lEntry.getInputs());
        }


        //Discriminator layers
        LEntry[] DislEntry = NeuralNetwork.discriminatorLayers();

        //Changing first inputs of first Vertex of discriminator
        graphBuilder.addVertex(
                DislEntry[0].getLayerName(),
                ((VertexEntry)DislEntry[0]).getVertex(),
                GenlEntry[GenlEntry.length - 2].getLayerName(), "Input");

        for (int i = 1; i < DislEntry.length; i++)
            graphBuilder.addLayer(
                    DislEntry[i].getLayerName(),
                    ((LayerEntry) DislEntry[i]).getLayer(),
                    DislEntry[i].getInputs()
            );

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
            network.getLayer(i).setMaskArray(discriminator.getLayer(i - genLayerCount).getMaskArray());
        }
    }

    /**
     * GAN builder, used as a starting point for creating a MultiLayerConfiguration or
     * ComputationGraphConfiguration.<br>
     */
    public static class Builder implements Cloneable {
        protected Supplier<ComputationGraph> generator;
        protected Supplier<ComputationGraph> discriminator;

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
        public GAN.Builder discriminator(Supplier<ComputationGraph> discriminator) {
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
        private final static INDArray REAL = Nd4j.ones(1,1,16,16);
        private final static INDArray FAKE = Nd4j.zeros(1,1,16,16);
    }
}