package org.inPainting.nn.dataSets;


import javafx.scene.image.Image;
import javafx.scene.image.PixelReader;
import javafx.scene.paint.Color;
import lombok.Getter;
import lombok.SneakyThrows;
import lombok.Synchronized;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.MultiDataSet;
import org.nd4j.linalg.dataset.api.MultiDataSetPreProcessor;
import org.nd4j.linalg.factory.Nd4j;
import org.inPainting.nn.GAN;

import java.io.FileInputStream;
import java.io.IOException;
import java.util.Random;

public final class  ImageMemoryDataSetIterator extends ImageDataSetIterator {

    private Random r = new Random();

    private MultiDataSet[] multiDataSets;

    @Getter
    private MultiDataSetPreProcessor preProcessor = null;

    private int pointer = 0;

    private int iterationsPerPicture = 20;


    public ImageMemoryDataSetIterator(int IterationsPerPicture, MultiDataSet[] multiDataSets){
        this(multiDataSets);
        this.iterationsPerPicture = IterationsPerPicture;
    }

    public ImageMemoryDataSetIterator(MultiDataSet[] multiDataSets){
        this.multiDataSets = multiDataSets;
        super.maxSize = (long) (multiDataSets.length - 1) * iterationsPerPicture;
    }

    public ImageMemoryDataSetIterator(MultiDataSet[] multiDataSets, int seed){
        this(multiDataSets);
        this.r = new Random(seed);
    }


    @SneakyThrows
    public ImageMemoryDataSetIterator(int IterationsPerPicture, FileEntry[] entries){
        this(entries);
        this.iterationsPerPicture = IterationsPerPicture;
        super.maxSize = (long) (multiDataSets.length - 1) * iterationsPerPicture;
    }

    @SneakyThrows
    public ImageMemoryDataSetIterator(FileEntry[] entries){
        this.multiDataSets = new MultiDataSet[entries.length];

        for (int i = 0; i < entries.length; i++)
            this.multiDataSets[i] = convertToDataSet(entries[i]);

        super.maxSize = (long) (multiDataSets.length - 1) * iterationsPerPicture;
    }

    @SneakyThrows
    public ImageMemoryDataSetIterator(FileEntry[] entries, int seed){
        this(entries);
        this.r = new Random(seed);
    }

    @Override
    @Synchronized
    public MultiDataSet next(int num) {
        return multiDataSets[num];
    }

    @Override
    public void setPreProcessor(MultiDataSetPreProcessor preProcessor) {
        for (MultiDataSet multiDataSet : multiDataSets)
            preProcessor.preProcess(multiDataSet);
        this.preProcessor = preProcessor;
    }

    @Override
    @Synchronized
    public MultiDataSet nextRandom(){
        return this.multiDataSets[this.r.nextInt(this.multiDataSets.length)];
    }

    @Override
    public boolean resetSupported() {
        return true;
    }

    @Override
    public boolean asyncSupported() {
        return true;
    }

    @Override
    @Synchronized
    public void reset() {
        this.pointer = 0;
        this.shuffle();
        System.gc();
    }

    @Override
    @Synchronized
    public void shuffle() {
        MultiDataSet[] ar = this.multiDataSets;
        for (int i = ar.length - 1; i > 0; i--) {
            int index = r.nextInt(i + 1);
            MultiDataSet a = ar[index];
            ar[index] = ar[i];
            ar[i] = a;
        }
        this.multiDataSets = ar;
    }

    @Override
    @Synchronized
    public boolean hasNext() {
        return (pointer < super.maxSize);
    }

    @Override
    @Synchronized
    public MultiDataSet next() {
        if (this.hasNext()){
            pointer++;
            return multiDataSets[(int)(pointer / iterationsPerPicture)];
        } else
            return multiDataSets[multiDataSets.length - 1];
    }

    @Override
    protected INDArray convertToRank4INDArrayOutput(Image inputImage) {

        assert inputImage != null;
        assert inputImage.getHeight() <= GAN._InputShape[0][2];
        assert inputImage.getWidth() <= GAN._InputShape[0][3];

        int width = (int) inputImage.getWidth();
        int height = (int) inputImage.getHeight();

        INDArray temp0 = Nd4j.zeros(1,3,height,width);
        PixelReader inputPR = inputImage.getPixelReader();

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                Color inputColor = inputPR.getColor(x, y);

                double fCr = scaleColor(inputColor.getRed());
                double fCg = scaleColor(inputColor.getGreen());
                double fCb = scaleColor(inputColor.getBlue());
                //double fCA = scaleColor(inputColor.getOpacity());

                temp0.putScalar(new int[]{0,0,y,x},fCr);
                temp0.putScalar(new int[]{0,1,y,x},fCg);
                temp0.putScalar(new int[]{0,2,y,x},fCb);
            }
        }
        return temp0;
    }

    @Override
    protected INDArray convertToRank4INDArrayInput(Image inputImage) {

        assert inputImage != null;

        assert inputImage.getHeight() <= GAN._InputShape[0][2];
        assert inputImage.getWidth() <= GAN._InputShape[0][3];

        int width = (int) inputImage.getWidth();
        int height = (int) inputImage.getHeight();


        INDArray temp0 = Nd4j.zeros(GAN._InputShape[0][0], GAN._InputShape[0][1],height,width);

        PixelReader inputPR = inputImage.getPixelReader();

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {

                Color inputColor = inputPR.getColor(x, y);

                double fCr = scaleColor(inputColor.getRed());
                double fCg = scaleColor(inputColor.getGreen());
                double fCb = scaleColor(inputColor.getBlue());

                temp0.putScalar(new int[]{0,0,y,x},fCr);
                temp0.putScalar(new int[]{0,1,y,x},fCg);
                temp0.putScalar(new int[]{0,2,y,x},fCb);
            }
        }
        return temp0;
    }

    @Override
    protected MultiDataSet convertToDataSet(FileEntry fileEntry) throws IOException {

        FileInputStream inputImageFileInputStream = new FileInputStream(fileEntry.getInput());
        FileInputStream expectedImageImageFileInputStream = new FileInputStream(fileEntry.getOutput());

        Image tempI0 = new Image(inputImageFileInputStream);
        Image tempI1 = new Image(expectedImageImageFileInputStream);

        if (tempI0.getWidth() != tempI1.getWidth() ||
                tempI0.getHeight() != tempI1.getHeight())
            throw new RuntimeException("Input and expected images have different sizes");

        INDArray temp1 = this.convertToRank4INDArrayInput(tempI0);
        INDArray temp2 = this.convertToRank4INDArrayOutput(tempI1);

        inputImageFileInputStream.close();
        expectedImageImageFileInputStream.close();

        MultiDataSet result = new MultiDataSet(
                new INDArray[] {
                        temp1,
                },
                new INDArray[] {
                        temp2
                }
        );

        if (preProcessor!=null)
            preProcessor.preProcess(result);

        return result;
    }
}
