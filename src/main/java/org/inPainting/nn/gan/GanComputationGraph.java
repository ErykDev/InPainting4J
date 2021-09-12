package org.inPainting.nn.gan;

import org.deeplearning4j.nn.api.FwdPassType;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.layers.IOutputLayer;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.graph.vertex.GraphVertex;
import org.deeplearning4j.nn.graph.vertex.impl.LayerVertex;
import org.deeplearning4j.nn.layers.FrozenLayerWithBackprop;
import org.deeplearning4j.nn.workspace.ArrayType;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.io.IOException;
import java.util.Collection;
import java.util.Map;

public class GanComputationGraph extends ComputationGraph {

    public GanComputationGraph(ComputationGraphConfiguration configuration) {
        super(configuration);
    }

    @Override
    public void computeGradientAndScore() {
        INDArray[] inputs = super.getInputs();
        INDArray[] inputMaskArrays = super.getInputMaskArrays();
        INDArray[] labelMaskArrays = super.getLabelMaskArrays();
        Collection<TrainingListener> trainingListeners = super.getListeners();


        synchronizeIterEpochCounts();

        LayerWorkspaceMgr workspaceMgr;
        if(configuration.getTrainingWorkspaceMode() == WorkspaceMode.NONE){
            workspaceMgr = LayerWorkspaceMgr.noWorkspaces();
        } else {
            workspaceMgr = LayerWorkspaceMgr.builder()
                    .with(ArrayType.ACTIVATIONS, WS_ALL_LAYERS_ACT, WS_ALL_LAYERS_ACT_CONFIG)
                    .with(ArrayType.INPUT, WS_ALL_LAYERS_ACT, WS_ALL_LAYERS_ACT_CONFIG)
                    .with(ArrayType.FF_WORKING_MEM, WS_LAYER_WORKING_MEM, WS_LAYER_WORKING_MEM_CONFIG)
                    .with(ArrayType.BP_WORKING_MEM, WS_LAYER_WORKING_MEM, WS_LAYER_WORKING_MEM_CONFIG)
                    .with(ArrayType.RNN_FF_LOOP_WORKING_MEM, WS_RNN_LOOP_WORKING_MEM, WS_RNN_LOOP_WORKING_MEM_CONFIG)
                    .with(ArrayType.RNN_BP_LOOP_WORKING_MEM, WS_RNN_LOOP_WORKING_MEM, WS_RNN_LOOP_WORKING_MEM_CONFIG)
                    //Note for updater working memory, we have the option to re-use WS_ALL_LAYERS_ACT or FF/BP_WORKING_MEM
                    // as these should be closed by the time updaters are executed
                    //Generally, WS_ALL_LAYERS_ACT will be the larger of the two, so we'll use this
                    .with(ArrayType.UPDATER_WORKING_MEM, WS_ALL_LAYERS_ACT, WS_ALL_LAYERS_ACT_CONFIG)
                    .build();
        }
        workspaceMgr.setHelperWorkspacePointers(helperWorkspaces);

        boolean tbptt = configuration.getBackpropType() == BackpropType.TruncatedBPTT;
        FwdPassType fwdType = (tbptt ? FwdPassType.RNN_ACTIVATE_WITH_STORED_STATE : FwdPassType.STANDARD);
        synchronizeIterEpochCounts();

        //Calculate activations (which are stored in each layer, and used in backprop)
        try(MemoryWorkspace wsAllActivations = workspaceMgr.notifyScopeEntered(ArrayType.ACTIVATIONS)) {
            Map<String, INDArray> activations = ffToLayerActivationsInWS(true, -1, getOutputLayerIndices(),
                    fwdType, tbptt, inputs, inputMaskArrays, labelMaskArrays, false);
            if (!trainingListeners.isEmpty()) {
                try (MemoryWorkspace workspace = Nd4j.getMemoryManager().scopeOutOfWorkspaces()) {
                    for (TrainingListener tl : trainingListeners) {
                        tl.onForwardPass(this, activations);
                    }
                }
            }
            calcBackpropGradients(false,false);

            workspaceMgr.assertCurrentWorkspace(ArrayType.ACTIVATIONS, null);

            //Score: sum of the scores for the various output layers...
            double r = calcRegularizationScore(true);

            score = 0.0;
            int outNum = 0;
            for (String s : configuration.getNetworkOutputs()) {
                GraphVertex gv = verticesMap.get(s);
                if(gv instanceof LayerVertex) {
                    //At this point: the input to the output layer might not be set on the layer itself - just the vertex
                    LayerVertex lv = (LayerVertex) gv;
                    if(!lv.isSetLayerInput()) {
                        lv.applyPreprocessorAndSetInput(workspaceMgr);
                    }
                }
                Layer vertexLayer = gv.getLayer();
                if (vertexLayer instanceof FrozenLayerWithBackprop) {
                    vertexLayer = ((FrozenLayerWithBackprop) vertexLayer).getInsideLayer();
                }
                vertexLayer.setMaskArray((labelMaskArrays == null) ? null : labelMaskArrays[outNum]);

                try(MemoryWorkspace ws = workspaceMgr.notifyScopeEntered(ArrayType.FF_WORKING_MEM)) {
                    switch (s){
                        case "DISCNNLoss":
                            score += ((IOutputLayer) vertexLayer).computeScore(r, true, workspaceMgr) * 100.0; //multiplying loss_weight of discriminator
                            break;
                        default:
                            score += ((IOutputLayer) vertexLayer).computeScore(r, true, workspaceMgr);
                    }
                }

                //Only want to add l1/l2 component once...
                r = 0.0;
                outNum++;
            }

            //Listeners
            if (!trainingListeners.isEmpty()) {
                try (MemoryWorkspace workspace = Nd4j.getMemoryManager().scopeOutOfWorkspaces()) {
                    for (TrainingListener tl : trainingListeners) {
                        tl.onBackwardPass(this);
                    }
                }
            }
        }

        for(GraphVertex gv : vertices){
            gv.clear();
        }
    }

    public static GanComputationGraph load(File f, boolean loadUpdater) throws IOException {
        return GanComputationGraphUtils.load(f, loadUpdater);
    }
}
