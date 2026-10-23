from PIL import Image,ImageDraw

from app.models.schemas import BoundingBox,DetectedElement
from app.modules.visual_groups import augment_visual_groups


def test_label_above_empty_field_stays_text_and_field_is_native():
    image=Image.new('RGB',(400,300),'white');draw=ImageDraw.Draw(image)
    draw.rounded_rectangle((40,85,340,125),4,outline='#506070',width=2)
    label=DetectedElement(element_id='email',element_type='input',text_content='Email address',bbox=BoundingBox(x=40,y=55,width=110,height=22))
    result=augment_visual_groups(image,[label])
    assert result[0].element_type=='text'
    fields=[element for element in result if element.element_type=='input']
    assert len(fields)==1
    assert fields[0].bbox.y>=80 and fields[0].bbox.width>250
    assert fields[0].attributes=={'aria-label':'Email address','type':'email'}


def test_plain_text_does_not_create_a_control():
    image=Image.new('RGB',(400,300),'white')
    text=DetectedElement(element_id='text',element_type='text',text_content='A decorative title',bbox=BoundingBox(x=40,y=55,width=180,height=22))
    assert augment_visual_groups(image,[text])==[text]


def test_original_text_anchor_is_retained_when_label_box_was_padded():
    image=Image.new('RGB',(400,300),'white');draw=ImageDraw.Draw(image)
    draw.rounded_rectangle((40,85,340,125),4,outline='#506070',width=2)
    label=DetectedElement(element_id='email',element_type='input',text_content='Email address',bbox=BoundingBox(x=28,y=43,width=134,height=45),attributes={'data-text-bbox':'[40,55,110,22]'})
    result=augment_visual_groups(image,[label])
    assert result[0].element_type=='text'
    assert result[0].bbox==BoundingBox(x=40,y=55,width=110,height=22)
