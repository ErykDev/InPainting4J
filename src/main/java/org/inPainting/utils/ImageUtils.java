package org.inPainting.utils;

import javafx.scene.image.Image;
import javafx.scene.image.PixelReader;
import javafx.scene.image.PixelWriter;
import javafx.scene.image.WritableImage;
import javafx.scene.paint.Color;
import org.datavec.image.loader.ImageLoader;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.MultiDataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.inPainting.nn.ImageDataSetIterator;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.net.MalformedURLException;
import java.util.Arrays;
import java.util.Random;

public final class ImageUtils {
    private final static int[] _NetInputShape = {1,4,256,256};

    private ImageUtils() {
        // This is intentionally empty
    }

    public static WritableImage emptyImage(Color color, int width, int height) {

        WritableImage writableImage = new WritableImage(width, height);
        PixelWriter pixelWriter = writableImage.getPixelWriter();

        for (int y = 0; y < height; y++)
            for (int x = 0; x < width; x++)
                pixelWriter.setColor(x, y, color);

        return writableImage;
    }

    public static WritableImage drawImage(INDArray data, int width, int height) {

        WritableImage writableImage = new WritableImage(width, height);
        PixelWriter pixelWriter = writableImage.getPixelWriter();

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                double r = data.getDouble(0,0,y,x);
                double g = data.getDouble(0,1,y,x);
                double b = data.getDouble(0,2,y,x);
                Color color = new Color(r, g, b, 1);
                pixelWriter.setColor(x, y, color);
            }
        }
        return writableImage;
    }



    public static INDArray convertToRank4INDArrayOutput(Image inputImage) {

        assert inputImage != null;
        assert inputImage.getHeight() <= ImageUtils._NetInputShape[2];
        assert inputImage.getWidth() <= ImageUtils._NetInputShape[3];

        int width = (int) inputImage.getWidth();
        int height = (int) inputImage.getHeight();

        int maskChannels = 1;

        INDArray fArray = Nd4j.zeros(ImageUtils._NetInputShape[0],(ImageUtils._NetInputShape[1] - maskChannels),height,width);
        PixelReader inputPR = inputImage.getPixelReader();

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                Color inputColor = inputPR.getColor(x, y);

                double fCr = scaleColor(inputColor.getRed());
                double fCg = scaleColor(inputColor.getGreen());
                double fCb = scaleColor(inputColor.getBlue());
                //double fCA = scaleColor(inputColor.getOpacity());

                fArray.putScalar(new int[]{0,0,y,x},fCr);
                fArray.putScalar(new int[]{0,1,y,x},fCg);
                fArray.putScalar(new int[]{0,2,y,x},fCb);
            }
        }
        return fArray;
    }

    public static INDArray convertToRank4INDArrayInput(Image inputImage, Image mask) {

        assert inputImage != null;
        assert mask != null;

        assert mask.getHeight() == inputImage.getHeight();
        assert mask.getWidth() == inputImage.getWidth();

        assert inputImage.getHeight() <= ImageUtils._NetInputShape[2];
        assert inputImage.getWidth() <= ImageUtils._NetInputShape[3];

        int width = (int) inputImage.getWidth();
        int height = (int) inputImage.getHeight();

        INDArray fArray = Nd4j.zeros(ImageUtils._NetInputShape[0],ImageUtils._NetInputShape[1],height,width);

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

                fArray.putScalar(new int[]{0,0,y,x},mB);

                fArray.putScalar(new int[]{0,1,y,x},fCr);
                fArray.putScalar(new int[]{0,2,y,x},fCg);
                fArray.putScalar(new int[]{0,3,y,x},fCb);
            }
        }
        return fArray;
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

    public static ImageDataSetIterator prepareData(){

        MultiDataSet[] res = new MultiDataSet[new File(ImageUtils.class.getResource("/data/256/inputs/").getFile()).listFiles().length/2];

        for (int i = 1; i < res.length + 1; i++) {
            try {
                res[i-1] = ImageUtils.convertToDataSet(
                        new File(ImageUtils.class.getResource("/data/256/inputs/input"+i+".png").getFile()),
                        new File(ImageUtils.class.getResource("/data/256/inputs/input"+i+"_mask.png").getFile()),
                        new File(ImageUtils.class.getResource("/data/256/expected/expected"+i+".png").getFile()));
            } catch (FileNotFoundException e) {
                e.printStackTrace();
            }
        }
        return new ImageDataSetIterator(res);
    }


    public static MultiDataSet convertToDataSet(File iImage, File iMascImage, File eImage) throws FileNotFoundException {

        Image inputImage = new Image(new FileInputStream(iImage));
        Image expectedImage = new Image(new FileInputStream(eImage));
        Image expectedImageMask = new Image(new FileInputStream(iMascImage));

        if (inputImage.getWidth() != expectedImage.getWidth() ||
                inputImage.getHeight() != expectedImage.getHeight())
            throw new RuntimeException("Input and expected images have different sizes");

        return new MultiDataSet(
                new INDArray[] {
                        ImageUtils.convertToRank4INDArrayInput(inputImage,expectedImageMask)
                },
                new INDArray[] {
                        ImageUtils.convertToRank4INDArrayOutput(expectedImage)
                }
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