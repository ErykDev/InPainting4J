package org.inPainting.nn.dataSets;

import javafx.scene.image.Image;
import javafx.scene.image.PixelReader;
import javafx.scene.paint.Color;
import lombok.Getter;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.MultiDataSet;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.io.IOException;

public abstract class ImageDataSetIterator implements MultiDataSetIterator {
    @Getter
    public long maxSize;

    public abstract MultiDataSet nextRandom();

    /**
     * shuffling Data in Iterator
     */
    public abstract void shuffle();

    /**
     * @param inputImage javafx.scene.image.Image to convert
     * @return INDArray of given Image (3 dimensions)
     */
    protected abstract INDArray convertToRank4INDArrayOutput(Image inputImage);

    /**
     * @param inputImage javafx.scene.image.Image to convert
     * @return INDArray of given Image (3 dimensions)
     */
    protected abstract INDArray convertToRank4INDArrayInput(Image inputImage);

    /**
     * @param inputImageMask javafx.scene.image.Image representing mask (brightness)
     * @return INDArray of given Mask (2 dimensions) representation of the mask
     */
    protected abstract INDArray convertToRank4INDArrayInputMask(Image inputImageMask);

    protected abstract MultiDataSet convertToDataSet(FileEntry fileEntry) throws IOException;
    public abstract MultiDataSet next();

    protected double scaleColor(double value) {
        return (value);
    }

    protected void loadImage(Image inputImage, INDArray output) {
        int width = (int) inputImage.getWidth();
        int height = (int) inputImage.getHeight();
        PixelReader inputPR = inputImage.getPixelReader();

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {

                Color inputColor = inputPR.getColor(x, y);

                double fCr = scaleColor(inputColor.getRed());
                double fCg = scaleColor(inputColor.getGreen());
                double fCb = scaleColor(inputColor.getBlue());

                output.putScalar(new int[]{0,0,y,x},fCr);
                output.putScalar(new int[]{0,1,y,x},fCg);
                output.putScalar(new int[]{0,2,y,x},fCb);
            }
        }
    }

    protected void loadMask(Image inputImageMask, INDArray output) {
        int width = (int) inputImageMask.getWidth();
        int height = (int) inputImageMask.getHeight();
        PixelReader inputPR = inputImageMask.getPixelReader();

        for (int y = 0; y < height; y++)
            for (int x = 0; x < width; x++) {
                Color inputColor = inputPR.getColor(x, y);
                double fBr = scaleColor(inputColor.getBrightness());

                output.putScalar(new int[]{0,0,y,x}, fBr);
            }
    }

    public static class FileEntry {
        @Getter
        private final File input;

        @Getter
        private final File input_mask;

        @Getter
        private final File output;

        public FileEntry(File input, File input_mask, File output){
            this.input = input;
            this.output = output;
            this.input_mask = input_mask;
        }
    }
}
