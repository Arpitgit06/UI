"""Product acceptance is independent of a successful build or screenshot replay."""
from app.modules.scene_plan import nodes, application_entry


def assess_reconstruction(layouts, visual):
    entry=layouts[application_entry(layouts)]
    elements=list(nodes(entry.root))[1:]
    controls=[n for n in elements if n.tag_hint in {'input','button','select','a'}]
    missing=[n.node_id for n in elements if n.text_content and
             n.text_content.strip().lower().startswith(('paste ', 'scan ', 'enter ', 'submit '))
             and n.tag_hint not in {'input','button','select','a'}]
    reasons=[]
    if not controls:reasons.append('Application screen has no native controls; preserved imagery is not an editable UI')
    if missing:reasons.append('Visible control cues remain static text')
    checks=visual.get('interaction_checks',[])
    if controls and not any(c.get('check')=='native_form_validation_and_local_submission' for c in checks):
        reasons.append('No native form submission was browser-tested')
    if visual.get('hybrid_status')!='passed':reasons.append('Native controls/artwork still differ from the source')
    if controls and not entry.vision_verifier_approved:
        reasons.append('Detected control semantics still require independent review')
    accepted=not reasons
    return {'status':'accepted' if accepted else 'preview_only' if not controls or missing else 'needs_review',
            'accepted':accepted,'scope':'reconstructed frontend only; backend behavior is not verified',
            'entry_state':entry.state_name,'native_controls':len(controls),
            'missing_control_nodes':missing,'reasons':reasons,
            'source_media_counts_as_ui_reconstruction':False}
