pub(crate) struct Frame {
    height: u16,
    width: u16,

    // pixel format? default to 8 bit luma per pixel right now
    data: Vec<u8>
}

impl Frame {
    pub fn get_height(self) -> u16 {
        self.height
    }

    pub fn get_width(self) -> u16 {
        self.width
    }

    pub fn get_data<'a>(self) -> &'a Vec<u8> {
        return &self.data
    }
}