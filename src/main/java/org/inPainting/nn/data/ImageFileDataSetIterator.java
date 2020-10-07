package org.inPainting.nn.data;


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

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.Random;

public final class ImageFileDataSetIterator extends ImageDataSetIterator {

    private Random r;

    private FileEntry[] fileEntries;

    @Getter
    private MultiDataSetPreProcessor preProcessor = null;

    private int pointer = 0;

    private int iterationsPerPicture = 20;

    private FileInputStream inputImageFileInputStream;
    private FileInputStream expectedImageImageFileInputStream;
    private FileInputStream expectedImageMaskImageFileInputStream;

    private INDArray temp0;
    private INDArray temp1;
    private INDArray temp2;
    private INDArray temp3;

    private Image tempI0;
    private Image tempI1;
    private Image tempI2;

    @Getter
    private int maxSize;

    public ImageFileDataSetIterator(int IterationsPerPicture, FileEntry[] fileEntries){
        this.iterationsPerPicture = IterationsPerPicture;
        this.fileEntries = fileEntries;
        this.maxSize = (fileEntries.length - 1) * IterationsPerPicture;
        this.r = new Random();

        this.initFirstSet();

        this.shuffle();
    }

    public ImageFileDataSetIterator(FileEntry[] fileEntries){
        this.fileEntries = fileEntries;
        this.maxSize = (fileEntries.length - 1) * iterationsPerPicture;
        this.r = new Random();

        this.initFirstSet();

        this.shuffle();
    }

    public ImageFileDataSetIterator(FileEntry[] fileEntries, int seed){
        this.fileEntries = fileEntries;
        this.maxSize = (fileEntries.length - 1) * iterationsPerPicture;
        this.r = new Random(seed);

        this.initFirstSet();

        this.shuffle();
    }

    @SneakyThrows
    @Override
    @Synchronized
    public MultiDataSet next(int num) {
        return this.convertToDataSet(fileEntries[num]);
    }

    @Override
    public void setPreProcessor(MultiDataSetPreProcessor preProcessor) {
        this.preProcessor = preProcessor;
    }

    @SneakyThrows
    @Synchronized
    public MultiDataSet nextRandom(){
        return this.convertToDataSet(this.fileEntries[this.r.nextInt(this.fileEntries.length)]);
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
        FileEntry[] ar = this.fileEntries;
        for (int i = ar.length - 1; i > 0; i--) {
            int index = r.nextInt(i + 1);
            FileEntry a = ar[index];
            ar[index] = ar[i];
            ar[i] = a;
        }
        this.fileEntries = ar;
    }

    @Override
    @Synchronized
    public boolean hasNext() {
        return (pointer < maxSize);
    }

    @SneakyThrows
    @Override
    @Synchronized
    public MultiDataSet next() {
        if (this.hasNext()){
            pointer++;
            //pointer same as before so no need to read data again
            if ((int)((pointer-1) / iterationsPerPicture) == (int)(pointer / iterationsPerPicture)){
                return new MultiDataSet(
                        new INDArray[] { temp1, temp3 },
                        new INDArray[] { temp2 }
                );
            } else
                return this.convertToDataSet(fileEntries[(int)(pointer / iterationsPerPicture)]);
        } else
            return this.convertToDataSet(fileEntries[fileEntries.length - 1]);
    }


    @SneakyThrows
    private void initFirstSet(){
        this.convertToDataSet(fileEntries[0]);
    }


    private INDArray convertToRank4INDArrayInput(Image inputImage) {

        assert inputImage != null;

        assert inputImage.getHeight() <= GAN._MergedNetInputShape[2];
        assert inputImage.getWidth() <= GAN._MergedNetInputShape[3];

        int width = (int) inputImage.getWidth();
        int height = (int) inputImage.getHeight();


        temp0 = Nd4j.zeros(GAN._MergedNetInputShape[0], GAN._MergedNetInputShape[1] - 1,height,width);

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

    private INDArray convertToRank4INDArrayMask(Image maskImage) {

        int width = (int) maskImage.getWidth();
        int height = (int) maskImage.getHeight();

        temp0 = Nd4j.zeros(1,1,height,width);
        PixelReader maskinputPR = maskImage.getPixelReader();

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                Color inputMaskColor = maskinputPR.getColor(x, y);

                double mB = scaleColor(inputMaskColor.getBrightness());

                temp0.putScalar(new int[]{0,0,y,x},mB);
            }
        }
        return temp0;
    }

    private INDArray convertToRank4INDArrayOutput(Image inputImage) {

        assert inputImage != null;
        assert inputImage.getHeight() <= GAN._MergedNetInputShape[2];
        assert inputImage.getWidth() <= GAN._MergedNetInputShape[3];

        int width = (int) inputImage.getWidth();
        int height = (int) inputImage.getHeight();

        temp0 = Nd4j.zeros(1,3,height,width);
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

    private MultiDataSet convertToDataSet(FileEntry fileEntry) throws IOException {

        inputImageFileInputStream = new FileInputStream(fileEntry.getInput());
        expectedImageImageFileInputStream = new FileInputStream(fileEntry.getOutput());
        expectedImageMaskImageFileInputStream = new FileInputStream(fileEntry.getMask());

        tempI0 = new Image(inputImageFileInputStream);
        tempI1 = new Image(expectedImageImageFileInputStream);
        tempI2 = new Image(expectedImageMaskImageFileInputStream);

        if (tempI0.getWidth() != tempI1.getWidth() ||
                tempI0.getHeight() != tempI1.getHeight())
            throw new RuntimeException("Input and expected images have different sizes");

        temp1 = this.convertToRank4INDArrayInput(tempI0);
        temp2 = this.convertToRank4INDArrayOutput(tempI1);
        temp3 = this.convertToRank4INDArrayMask(tempI2);

        inputImageFileInputStream.close();
        expectedImageImageFileInputStream.close();
        expectedImageMaskImageFileInputStream.close();

        inputImageFileInputStream = null;
        expectedImageImageFileInputStream = null;
        expectedImageMaskImageFileInputStream = null;

        MultiDataSet result = new MultiDataSet(
                new INDArray[] { temp1, temp3 },
                new INDArray[] { temp2 }
        );

        if (preProcessor!=null) {
            preProcessor.preProcess(result);
        }
        return result;
    }

    private double scaleColor(double value) {
        return (value);
    }

    public static class FileEntry {

        @Getter
        private File input;
        @Getter
        private File output;
        @Getter
        private File mask;

        public FileEntry(File input, File output, File mask){
            this.input = input;
            this.output = output;
            this.mask = mask;
        }

        @Override
        protected void finalize() throws Throwable {
            input = null;
            output = null;
            mask = null;
        }
    }
}
