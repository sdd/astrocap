extern crate opencv;

use opencv::prelude::*;

pub fn empty(mat: &Mat) -> opencv::Result<bool> {
    let size = mat.size()?;
    Ok(size.width == 0 && size.height == 0)
}
