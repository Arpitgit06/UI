"""Shared native primitives emitted once per exported project."""
MEASURED_COMPONENTS = r'''import React,{useState} from 'react';
export function MeasuredPanel({children,...props}) {
  return <section {...props} data-component="measured-panel">{children}</section>;
}
export function MeasuredForm({children,...props}) {
  const [status,setStatus]=useState('');
  return <form {...props} data-component="measured-form" onSubmit={event=>{
    event.preventDefault();
    if(!event.currentTarget.reportValidity())return;
    const values=Object.fromEntries(new FormData(event.currentTarget));
    event.currentTarget.dispatchEvent(new CustomEvent('omniui:submit',{bubbles:true,detail:{values}}));
    setStatus('Form interaction previewed.');
  }}>{children}{status&&<output role="status" className="omniui-form-status">{status}</output>}</form>;
}
'''
