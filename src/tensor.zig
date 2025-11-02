const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;

const root = @import("root.zig");
const array_like = root.array_like;

/// A multidimensional array backed up by a slice allocated by `ArrayList(T)`. deinitialize with `Tensor(T, shape).deinit`
pub fn Tensor(T: type, shape_: []const usize) type {
    return struct {
        // TYPES

        const Self = @This();

        /// The `type` of values of this `Tensor`
        pub const ElementType = T;
        /// A type alias for an array of `rank` `usize`s
        pub const Index = [rank]usize;
        /// An iterator that will go over each `Index` in a dimension0 major order (last dimension changes the fastest and first dimension changes the slowest).
        pub const IndexIterator = struct {
            /// the current serialized index that this iterator is going to yield next
            current: usize = 0,
            /// returns the next `Index` in this iterator or `null` if the iterator is empty
            pub fn next(self: *@This()) ?Index {
                const index = deserializeIndex(self.current) catch return null;
                self.current += 1;
                return index;
            }
        };

        // FIELDS

        /// `T` values in dimension0 major order
        elements: []ElementType,

        // CONSTANTS

        /// The rank of a `Tensor` is the number of dimensions
        pub const rank = shape.len;
        /// The length of each dimension of this `Tensor`
        pub const shape = shape_;
        /// the total number of elements in this `Tensor`
        pub const number_of_elements = array_like.product(shape);

        // FUNCTIONS

        /// allocates memory needed to store `number_of_elements` `T` values.
        /// calls `f` for at each index to determine the value of the `Tensor` there
        pub fn initFromFunction(allocator: Allocator, elementFunction: fn (*const Index) anyerror!ElementType) !Self {
            var elements = try ArrayList(ElementType).initCapacity(allocator, number_of_elements);
            var index_iterator = indexes();
            while (index_iterator.next()) |index|
                try elements.append(allocator, try elementFunction(&index));
            return .{ .elements = try elements.toOwnedSlice(allocator) };
        }
        /// allocates memory needed to store `number_of_elements` `T`.
        /// all values are set to zero
        pub fn initZeros(allocator: Allocator) !Self {
            return Self.initFromFunction(
                allocator,
                struct {
                    pub fn f(_: *const Index) anyerror!ElementType {
                        return 0;
                    }
                }.f,
            );
        }
        /// `Allocator.free`s `self.elements` and sets `self` to `undefined`
        pub fn deinit(self: *Self, allocator: Allocator) void {
            allocator.free(self.elements);
            self.* = undefined;
        }

        /// Convert a `Index` to a `usize` (dimension0 major)
        pub fn serializeIndex(index: *const Index) error{IndexOutOfBounds}!usize {
            var serialized_index: usize = 0;
            var stride: usize = 1;

            var dimension = rank;
            while (dimension > 0) {
                dimension -= 1;
                if (index[dimension] >= shape[dimension]) return error.IndexOutOfBounds;
                serialized_index += index[dimension] * stride;
                stride *= shape[dimension];
            }

            return serialized_index;
        }

        /// Convert a `usize` (dimension0 major) to a `Index`
        pub fn deserializeIndex(index: usize) error{IndexOutOfBounds}!Index {
            if (index >= number_of_elements) return error.IndexOutOfBounds;

            var deserialized_index: Index = undefined;
            var remaining = index;

            var dimension = rank;
            while (dimension > 0) {
                dimension -= 1;
                deserialized_index[dimension] = remaining % shape[dimension];
                remaining /= shape[dimension];
                if (deserialized_index[dimension] >= shape[dimension]) return error.IndexOutOfBounds;
            }

            if (remaining != 0) return error.IndexOutOfBounds;

            return deserialized_index;
        }

        /// returns an a immutable pointer to a specific element of this `Tensor`
        pub fn get(self: *const Self, index: *const Index) error{IndexOutOfBounds}!*const ElementType {
            return &self.elements.items[try serializeIndex(index)];
        }

        /// returns an a pointer to a specific element of this `Tensor`
        pub fn set(self: *Self, index: *const Index, value: ElementType) error{IndexOutOfBounds}!void {
            self.elements.items[try serializeIndex(index)] = value;
        }

        /// Returns an iterator that will go over each `Index` in a dimension0 major order (last dimension changes the fastest and first dimension changes the slowest).
        /// call `IndexIterator.next` to consume the next element of the iterator. `IndexIterator.next` will return null when there are no indexes left
        pub fn indexes() IndexIterator {
            return .{};
        }
    };
}

// TESTS

const test_allocator = std.testing.allocator;
const expect = std.testing.expect;

test "serialize index" {
    const ExampleTensor = Tensor(f32, &.{ 7, 1, 12 });
    var expected_serialized_index: usize = 0;
    var indexes = ExampleTensor.indexes();
    while (indexes.next()) |deserialized_index| : (expected_serialized_index += 1) {
        const actual_serialized_index = try ExampleTensor.serializeIndex(&deserialized_index);
        try expect(actual_serialized_index == expected_serialized_index);
    }
}

test "deserialize index" {
    const ExampleTensor = Tensor(f32, &.{ 13, 3, 9 });
    var serialized_index: usize = 0;
    var deserialized_indexes = ExampleTensor.indexes();
    while (deserialized_indexes.next()) |expected_deserialized_index| : (serialized_index += 1) {
        const actual_deserialized_index = try ExampleTensor.deserializeIndex(serialized_index);
        try expect(array_like.equal(actual_deserialized_index, expected_deserialized_index));
    }
}

test "product" {
    const factorial = struct {
        fn f(x: anytype) @TypeOf(x) {
            return if (x == 0) 1 else f(x - 1) * x;
        }
    }.f;
    const one_to_ten = [_]usize{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    const empty_array = [_]f32{};
    try expect(factorial(10) == array_like.product(&one_to_ten));
    try expect(factorial(0.0) == array_like.product(empty_array));
}

test "from function" {
    const ExampleTensor = Tensor(f32, &.{ 3, 3 });
    var actual_identity = try ExampleTensor.initFromFunction(
        test_allocator,
        struct {
            pub fn f(index: *const ExampleTensor.Index) anyerror!f32 {
                return if (array_like.allValuesEqual(index)) 1.0 else 0.0;
            }
        }.f,
    );
    defer actual_identity.deinit(test_allocator);
    const expected_identity = [ExampleTensor.number_of_elements]f32{ 1, 0, 0, 0, 1, 0, 0, 0, 1 };
    try expect(std.mem.eql(f32, actual_identity.elements, &expected_identity));
}

test "zeros" {
    const ExampleTensor = Tensor(f32, &.{ 3, 3 });
    var actual_zeros = try ExampleTensor.initZeros(test_allocator);
    defer actual_zeros.deinit(test_allocator);
    const expected_zeros = [ExampleTensor.number_of_elements]f32{
        0, 0, 0,
        0, 0, 0,
        0, 0, 0,
    };
    try expect(std.mem.eql(f32, actual_zeros.elements, &expected_zeros));
}

test "constants" {
    const ExampleTensor = Tensor(f32, &.{ 3, 3 });
    try expect(ExampleTensor.rank == 2);
    try expect(array_like.equal(ExampleTensor.shape, [_]usize{ 3, 3 }));
    try expect(ExampleTensor.number_of_elements == 9);
}
