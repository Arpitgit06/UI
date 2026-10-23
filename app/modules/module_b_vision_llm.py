"""Compatibility entry points; the legacy second VLM has been retired."""
async def analyze_state_fallback(state,cuda_device_index=None):
    result=await analyze_states_fallback([state],cuda_device_index)
    return result[0] if result else []

async def analyze_states_fallback(states,cuda_device_index=None):
    from app.modules.module_b_spatial_vision import analyze_states
    elements,_=await analyze_states(states,cuda_device_index=cuda_device_index)
    return elements
