package org.inPainting.utils;

import javafx.scene.image.Image;
import javafx.scene.image.PixelReader;
import javafx.scene.image.PixelWriter;
import javafx.scene.image.WritableImage;
import javafx.scene.paint.Color;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.MultiDataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.inPainting.nn.data.FileEntry;
import org.inPainting.nn.data.ImageFileDataSetIterator;
import org.inPainting.nn.data.ImageMemoryDataSetIterator;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;

public final class ImageLoader {

    private WritableImage writableTemp;

    private FileInputStream inputImageFileInputStreamTemp;
    private FileInputStream expectedImageImageFileInputStreamTemp;
    private FileInputStream expectedImageMaskImageFileInputStreamTemp;

    private Image inputImageTemp;
    private Image expectedImageTemp;
    private Image expectedImageMaskTemp;

    private INDArray temp0;
    private INDArray temp1;
    private INDArray temp2;

    private int[] netInputShape = {1,4,256,256};

    public ImageLoader(int... neuralNetworkInputShape) {
        assert neuralNetworkInputShape.length == 4;

        this.netInputShape = neuralNetworkInputShape;
    }

    public WritableImage emptyImage(Color color, int width, int height) {

        writableTemp = new WritableImage(width, height);
        PixelWriter pixelWriter = writableTemp.getPixelWriter();

        for (int y = 0; y < height; y++)
            for (int x = 0; x < width; x++)
                pixelWriter.setColor(x, y, color);

        return writableTemp;
    }

    public WritableImage drawImage(INDArray data, int width, int height) {

        writableTemp = new WritableImage(width, height);
        PixelWriter pixelWriter = writableTemp.getPixelWriter();

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                double r = data.getDouble(0,0,y,x);
                double g = data.getDouble(0,1,y,x);
                double b = data.getDouble(0,2,y,x);
                Color color = new Color(r, g, b, 1);
                pixelWriter.setColor(x, y, color);
            }
        }
        return writableTemp;
    }



    public INDArray convertToRank4INDArrayOutput(Image inputImage) {

        assert inputImage != null;
        assert inputImage.getHeight() <= netInputShape[2];
        assert inputImage.getWidth() <= netInputShape[3];

        int width = (int) inputImage.getWidth();
        int height = (int) inputImage.getHeight();

        int maskChannels = 1;

        temp0 = Nd4j.zeros(netInputShape[0],(netInputShape[1] - maskChannels),height,width);
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

    public INDArray convertToRank4INDArrayInput(Image inputImage, Image mask) {

        assert inputImage != null;
        assert mask != null;

        assert mask.getHeight() == inputImage.getHeight();
        assert mask.getWidth() == inputImage.getWidth();

        assert inputImage.getHeight() <= netInputShape[2];
        assert inputImage.getWidth() <= netInputShape[3];

        int width = (int) inputImage.getWidth();
        int height = (int) inputImage.getHeight();

        temp0 = Nd4j.zeros(netInputShape[0], netInputShape[1],height,width);

        PixelReader inputPR = inputImage.getPixelReader();
        PixelReader maskinputPR = mask.getPixelReader();

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {

                Color inputColor = inputPR.getColor(x, y);
                Color inputMaskColor = maskinputPR.getColor(x, y);

                double fCr = scaleColor(inputColor.getRed());
                double fCg = scaleColor(inputColor.getGreen());
                double fCb = scaleColor(inputColor.getBlue());

                double mB = scaleColor(inputMaskColor.getBrightness());

                temp0.putScalar(new int[]{0,0,y,x},mB);

                temp0.putScalar(new int[]{0,1,y,x},fCr);
                temp0.putScalar(new int[]{0,2,y,x},fCg);
                temp0.putScalar(new int[]{0,3,y,x},fCb);
            }
        }
        return temp0;
    }


    public static INDArray mergeImagesByMask(INDArray ImageWithMask, INDArray Image, int width, int height) {

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                if (ImageWithMask.getDouble(0,0,y,x) == 0.0D){
                    Image.putScalar(new int[]{0,0,y,x},ImageWithMask.getDouble(0,1,y,x));
                    Image.putScalar(new int[]{0,1,y,x},ImageWithMask.getDouble(0,2,y,x));
                    Image.putScalar(new int[]{0,2,y,x},ImageWithMask.getDouble(0,3,y,x));
                }
            }
        }
        return Image;
    }

    public ImageMemoryDataSetIterator prepareInMemoryData(){

        MultiDataSet[] res = new MultiDataSet[new File(ImageLoader.class.getResource("/data/256/inputs/").getFile()).listFiles().length/2];

        for (int i = 1; i < res.length + 1; i++) {
            try {
                res[i-1] = this.convertToDataSet(
                        new File(ImageLoader.class.getResource("/data/256/inputs/input"+i+".png").getFile()),
                        new File(ImageLoader.class.getResource("/data/256/inputs/input"+i+"_mask.png").getFile()),
                        new File(ImageLoader.class.getResource("/data/256/expected/expected"+i+".png").getFile()));
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        return new ImageMemoryDataSetIterator(res);
    }

    public ImageFileDataSetIterator prepareInFileData(){

        FileEntry[] res = new FileEntry[new File(ImageLoader.class.getResource("/data/256/inputs/").getFile()).listFiles().length/2];

        for (int i = 1; i < res.length + 1; i++) {
            res[i-1] = new FileEntry(
                    new File(ImageLoader.class.getResource("/data/256/inputs/input"+i+".png").getFile()),
                    new File(ImageLoader.class.getResource("/data/256/expected/expected"+i+".png").getFile()),
                    new File(ImageLoader.class.getResource("/data/256/inputs/input"+i+"_mask.png").getFile())
            );
        }
        return new ImageFileDataSetIterator(res);
    }

    public MultiDataSet convertToDataSet(File iImage, File iMascImage, File eImage) throws IOException {

        inputImageFileInputStreamTemp = new FileInputStream(iImage);
        expectedImageImageFileInputStreamTemp = new FileInputStream(eImage);
        expectedImageMaskImageFileInputStreamTemp = new FileInputStream(iMascImage);

        inputImageTemp = new Image(inputImageFileInputStreamTemp);
        expectedImageTemp = new Image(expectedImageImageFileInputStreamTemp);
        expectedImageMaskTemp = new Image(expectedImageMaskImageFileInputStreamTemp);

        if (inputImageTemp.getWidth() != expectedImageTemp.getWidth() ||
                inputImageTemp.getHeight() != expectedImageTemp.getHeight())
            throw new RuntimeException("Input and expected images have different sizes");

        temp1 = this.convertToRank4INDArrayInput(inputImageTemp,expectedImageMaskTemp);
        temp2 = this.convertToRank4INDArrayOutput(expectedImageTemp);

        inputImageFileInputStreamTemp.close();
        expectedImageImageFileInputStreamTemp.close();
        expectedImageMaskImageFileInputStreamTemp.close();

        return new MultiDataSet(
                new INDArray[] { temp1 },
                new INDArray[] { temp2 }
        );
    }

    public MultiDataSet convertToDataSet(FileEntry fileEntry) throws IOException {

        inputImageFileInputStreamTemp = new FileInputStream(fileEntry.getInput());
        expectedImageImageFileInputStreamTemp = new FileInputStream(fileEntry.getOutput());
        expectedImageMaskImageFileInputStreamTemp = new FileInputStream(fileEntry.getMask());

        inputImageTemp = new Image(inputImageFileInputStreamTemp);
        expectedImageTemp = new Image(expectedImageImageFileInputStreamTemp);
        expectedImageMaskTemp = new Image(expectedImageMaskImageFileInputStreamTemp);

        if (inputImageTemp.getWidth() != expectedImageTemp.getWidth() ||
                inputImageTemp.getHeight() != expectedImageTemp.getHeight())
            throw new RuntimeException("Input and expected images have different sizes");

        temp1 = this.convertToRank4INDArrayInput(inputImageTemp,expectedImageMaskTemp);
        temp2 = this.convertToRank4INDArrayOutput(expectedImageTemp);

        inputImageFileInputStreamTemp.close();
        expectedImageImageFileInputStreamTemp.close();
        expectedImageMaskImageFileInputStreamTemp.close();

        return new MultiDataSet(
                new INDArray[] { temp1 },
                new INDArray[] { temp2 }
        );
    }

    private static double scaleColor(double value) {
        return (value);
    }

    private static double scale(int value, int rangeSize) {
        return scale(value, rangeSize, 1.0d);
    }

    @SuppressWarnings("SameParameterValue")
    private static double scale(int value, int rangeSize, double targetRange) {
        return (targetRange / (double) rangeSize) * ((double) value) - targetRange * 0.5;
    }

    private static double trimToRange0to1(double value) {
        return trimToRange(value, 0.0, 1.0);
    }

    @SuppressWarnings("SameParameterValue")
    private static double trimToRange(double value, double min, double max) {
        return Math.max(Math.min(value, max), min);
    }
}