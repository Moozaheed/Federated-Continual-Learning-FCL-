# 🔍 Codebase Analysis & Feedback Report - Update April 28, 2026
**Branch**: `advance-run` | **Status**: 🟢 Ready for Run with Minor Technical Fixes

---

## ✅ RESOLVED ISSUES (From Previous Report)
The following critical issues from the previous review have been successfully implemented:
- ✅ **DER++ Parameters Added**: `alpha` and `beta` are now present in `DERConfig`.
- ✅ **Multimodal Utilities**: `load_multimodal_data` and `fit_multimodal` added to `utils.py`.
- ✅ **Feature Extraction**: `FTTransformer.extract_features()` implemented and used in fusion logic.
- ✅ **MobileNetV3 Handling**: Dynamic feature dimension detection implemented in `MobileNetExtractor`.
- ✅ **DERBuffer Robustness**: Reservoir sampling and input validation added to `DERBuffer`.
- ✅ **Advance Planning**: `advance_planning.md` is now a comprehensive 12-section research blueprint.

---

## 🔴 NEW CRITICAL ISSUES (Bugs found in latest implementation)

### Issue #1: `DERConfig` Attribute Mismatch (AttributeError)
**File**: `code/config.py` (Lines ~88-115)  
**Problem**: Copy-pasted aliases in `DERConfig` reference attributes that don't exist in that class.
```python
@property
def num_numerical_features(self) -> int:
    return self.input_dim  # ❌ ERROR: DERConfig has no attribute 'input_dim'
```
**Impact**: Any code accessing these properties on the `config.der` object will crash.

### Issue #2: `DERBuffer` Method Mismatch in `utils.py`
**File**: `code/utils.py` (Line ~200)  
**Problem**: `fit_multimodal` calls a non-existent method `get_batch`.
```python
# In fit_multimodal:
b_imgs, b_tabs, b_labels, b_logits = der_buffer.get_batch(imgs.size(0)) # ❌
```
**Expected**: `DERBuffer` defines the method as `sample_batch(batch_size)`.

### Issue #3: Severe Code Duplication & Incompleteness in `utils.py`
**File**: `code/utils.py`  
**Problem**: The following functions are implemented **twice** in the same file:
1. `load_multimodal_data` (Lines ~100 and ~450)
2. `create_multimodal_data_loaders` (Lines ~118 and ~504)
3. `fit_multimodal` (Lines ~135 and ~540)
**Issue**: The second implementation of `fit_multimodal` (Line ~540) is more complete but still contains a `# truncated for brevity` comment and ends abruptly.
**Impact**: Maintenance nightmare and unpredictable behavior depending on which version Python executes.

---

## 🟠 HIGH PRIORITY ISSUES

### Issue #4: `get_param_count()` Breaking Change
**File**: `code/model.py` (Line ~285)  
**Problem**: Method changed from returning a single `int` to a `Dict[str, int]`.
**Notebook Context**:
```python
params = model.get_param_count()  # Notebook expects an int for summary prints
```
**Impact**: Will cause `TypeError` or incorrect formatting in existing Jupyter notebook cells.

### Issue #5: BatchNorm Stability
**File**: `code/multimodal/fusion.py` (Line ~26)  
**Problem**: `nn.BatchNorm1d` is used in `fusion_mlp`.
**Risk**: In federated scenarios where a client might have a very small batch (e.g., `batch_size=1`), BatchNorm will fail during training. 
**Recommendation**: Use `LayerNorm` or add a check for `batch_size > 1`.

---

## 🟡 MEDIUM/LOW PRIORITY

### Issue #6: `torchvision` Pretrained Weights
**File**: `code/multimodal/image_extractor.py`  
**Note**: `pretrained=True` requires an internet connection to download weights on the first run. For offline hospital environments, a local path configuration should be added.

### Issue #7: Redundant Metric Keys
**File**: `code/utils.py`  
**Note**: `compute_metrics` returns both `roc_auc` and `auc_roc` with identical values. Harmless but redundant.

---

## 📋 Action Items for Final Polish

1. [ ] **Fix `config.py`**: Remove or fix aliases in `DERConfig`.
2. [ ] **Fix `utils.py`**: 
    - Consolidate duplicated functions.
    - Change `get_batch` to `sample_batch`.
    - Complete the `fit_multimodal` implementation.
3. [ ] **Fix `model.py`**: Ensure `get_param_count()` or an alias returns an `int` for notebook compatibility.
4. [ ] **Update Notebook**: Ensure it uses the new multimodal data loaders if performing the "Power Run".

---

**Summary**: The architecture is excellent and 95% complete. Resolving the `get_batch` and `config` attribute bugs will make the system fully operational for the advanced multimodal run.
