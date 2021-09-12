package org.inPainting.gan;

import lombok.NonNull;
import org.apache.commons.io.IOUtils;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.nd4j.common.primitives.Pair;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.factory.Nd4j;

import java.io.*;
import java.util.HashMap;
import java.util.Map;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;

public class GanComputationGraphUtils {
    private GanComputationGraphUtils(){};

    private static void checkInputStream(InputStream inputStream) throws IOException {
        //available method can return 0 in some cases: https://github.com/deeplearning4j/deeplearning4j/issues/4887
        int available;
        try{
            //InputStream.available(): A subclass' implementation of this method may choose to throw an IOException
            // if this input stream has been closed by invoking the close() method.
            available = inputStream.available();
        } catch (IOException e){
            throw new IOException("Cannot read from stream: stream may have been closed or is attempting to be read from" +
                    "multiple times?", e);
        }
        if(available <= 0){
            throw new IOException("Cannot read from stream: stream may have been closed or is attempting to be read from" +
                    "multiple times?");
        }
    }

    private static Map<String, byte[]> loadZipData(InputStream is) throws IOException {
        Map<String, byte[]> result = new HashMap<>();
        try (final ZipInputStream zis = new ZipInputStream(is)) {
            while (true) {
                final ZipEntry zipEntry = zis.getNextEntry();
                if (zipEntry == null)
                    break;
                if(zipEntry.isDirectory() || zipEntry.getSize() > Integer.MAX_VALUE)
                    throw new IllegalArgumentException();

                final int size = (int) (zipEntry.getSize());
                final byte[] data;
                if (size >= 0) { // known size
                    data = IOUtils.readFully(zis, size);
                }
                else { // unknown size
                    final ByteArrayOutputStream bout = new ByteArrayOutputStream();
                    IOUtils.copy(zis, bout);
                    data = bout.toByteArray();
                }
                result.put(zipEntry.getName(), data);
            }
        }
        return result;
    }

    public static final String UPDATER_BIN = "updaterState.bin";
    public static final String NORMALIZER_BIN = "normalizer.bin";
    public static final String CONFIGURATION_JSON = "configuration.json";
    public static final String COEFFICIENTS_BIN = "coefficients.bin";
    public static final String NO_PARAMS_MARKER = "noParams.marker";
    public static final String PREPROCESSOR_BIN = "preprocessor.bin";

    private static Pair<GanComputationGraph,Map<String,byte[]>> restoreComputationGraphHelper(@NonNull InputStream is, boolean loadUpdater)
            throws IOException {
        checkInputStream(is);

        Map<String, byte[]> files = loadZipData(is);

        boolean gotConfig = false;
        boolean gotCoefficients = false;
        boolean gotUpdaterState = false;
        boolean gotPreProcessor = false;

        String json = "";
        INDArray params = null;
        INDArray updaterState = null;
        DataSetPreProcessor preProcessor = null;


        byte[] config = files.get(CONFIGURATION_JSON);
        if (config != null) {
            //restoring configuration

            InputStream stream = new ByteArrayInputStream(config);
            BufferedReader reader = new BufferedReader(new InputStreamReader(stream));
            String line = "";
            StringBuilder js = new StringBuilder();
            while ((line = reader.readLine()) != null) {
                js.append(line).append("\n");
            }
            json = js.toString();

            reader.close();
            stream.close();
            gotConfig = true;
        }


        byte[] coefficients = files.get(COEFFICIENTS_BIN);
        if (coefficients != null) {
            if(coefficients.length > 0) {
                InputStream stream = new ByteArrayInputStream(coefficients);
                DataInputStream dis = new DataInputStream(stream);
                params = Nd4j.read(dis);

                dis.close();
                gotCoefficients = true;
            } else {
                byte[] noParamsMarker = files.get(NO_PARAMS_MARKER);
                gotCoefficients = (noParamsMarker != null);
            }
        }


        if (loadUpdater) {
            byte[] updaterStateEntry = files.get(UPDATER_BIN);
            if (updaterStateEntry != null) {
                InputStream stream = new ByteArrayInputStream(updaterStateEntry);
                DataInputStream dis = new DataInputStream(stream);
                updaterState = Nd4j.read(dis);

                dis.close();
                gotUpdaterState = true;
            }
        }

        byte[] prep = files.get(PREPROCESSOR_BIN);
        if (prep != null) {
            InputStream stream = new ByteArrayInputStream(prep);
            ObjectInputStream ois = new ObjectInputStream(stream);

            try {
                preProcessor = (DataSetPreProcessor) ois.readObject();
            } catch (ClassNotFoundException e) {
                throw new RuntimeException(e);
            }

            gotPreProcessor = true;
        }


        if (gotConfig && gotCoefficients) {
            ComputationGraphConfiguration confFromJson;
            try{
                confFromJson = ComputationGraphConfiguration.fromJson(json);
                if(confFromJson.getNetworkInputs() == null && (confFromJson.getVertices() == null || confFromJson.getVertices().size() == 0)){
                    //May be deserialized correctly, but mostly with null fields
                    throw new RuntimeException("Invalid JSON - not a ComputationGraphConfiguration");
                }
            } catch (Exception e){
                if(e.getMessage() != null && e.getMessage().contains("registerLegacyCustomClassesForJSON")){
                    throw e;
                }
                try{
                    MultiLayerConfiguration.fromJson(json);
                } catch (Exception e2){
                    //Invalid, and not a compgraph
                    throw new RuntimeException("Error deserializing JSON ComputationGraphConfiguration. Saved model JSON is" +
                            " not a valid ComputationGraphConfiguration", e);
                }
                throw new RuntimeException("Error deserializing JSON ComputationGraphConfiguration. Saved model appears to be " +
                        "a MultiLayerNetwork - use ModelSerializer.restoreMultiLayerNetwork instead");
            }

            //Handle legacy config - no network DataType in config, in beta3 or earlier
            if(params != null)
                confFromJson.setDataType(params.dataType());

            GanComputationGraph cg = new GanComputationGraph(confFromJson);
            cg.init(params, false);


            if (gotUpdaterState && updaterState != null) {
                cg.getUpdater().setStateViewArray(updaterState);
            }
            return new Pair<>(cg, null);
        } else
            throw new IllegalStateException("Model wasnt found within file: gotConfig: [" + gotConfig
                    + "], gotCoefficients: [" + gotCoefficients + "], gotUpdater: [" + gotUpdaterState + "]");
    }

    public static GanComputationGraph load(File f, boolean loadUpdater) throws IOException {
        return restoreComputationGraphHelper(new FileInputStream(f), loadUpdater).getFirst();
    }
}
