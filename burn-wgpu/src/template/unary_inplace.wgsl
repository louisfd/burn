@group(0)
@binding(0)
var<storage, read_write> input: array<elem>;

@compute
@workgroup_size(WORKGROUP_SIZE_X, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    BODY
}
